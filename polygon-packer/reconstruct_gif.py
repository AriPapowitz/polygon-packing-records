"""Reconstruct packing coordinates from an Erich Friedman Packing Center GIF.

The site GIFs are uniform: colored polygon fills with thin dark outlines on a
white background. Each inner polygon is one connected fill component (the dark
outline separates neighbors), so:
  centroid of component      -> polygon center
  nsi-fold phase of pixels   -> polygon orientation (mod 2*pi/nsi)
  convex hull of all ink     -> container corners -> similarity transform

Output JSON matches packer/validate conventions (container vertex at angle 0,
circumradius S, unit inner circumradius). The claimed table size s fixes the
scale; --inflate adds slack so pixel noise stays feasible — polish/squeeze it
back down with validate_packing.py or drop_one.py afterwards.

Usage:
    python reconstruct_gif.py 45.gif 45 3 5 3.51261 --out 45_recon.json
"""

import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser()
parser.add_argument("gif")
parser.add_argument("n", type=int)
parser.add_argument("nsi", type=int)
parser.add_argument("nsc", type=int)
parser.add_argument("s_claim", type=float, help="table value s (container side / inner side)")
parser.add_argument("--out", default=None)
parser.add_argument("--inflate", type=float, default=0.005,
                    help="relative container growth to absorb pixel noise")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

img = np.asarray(Image.open(args.gif).convert("RGB")).astype(int)
H, W, _ = img.shape
R, G, B = img[..., 0], img[..., 1], img[..., 2]

# fill = bright colored pixels (site fills are tinted, lines dark, background
# white; the >=200 brightness floor drops line-antialiasing blends that would
# otherwise bridge adjacent fills). Some pages use flat gray fills instead —
# fall back to a tight band around the dominant non-white shade.
sat = img.max(-1) - img.min(-1)
fill = (sat >= 15) & (img.max(-1) >= 200)
ink = ~((R > 235) & (G > 235) & (B > 235))          # anything not white-ish
if fill.sum() < 100:
    vals, counts = np.unique(img.max(-1)[ink & (sat < 10)], return_counts=True)
    shade = int(vals[counts.argmax()])
    fill = (sat < 8) & (np.abs(img.max(-1) - shade) <= 6)
    if fill.sum() < 100 or fill.sum() < 0.3 * ink.sum():
        # pastel style: fill = pixels near the dominant non-white RGB color
        pix = img[ink]
        dom, cnt = np.unique(pix.reshape(-1, 3), axis=0, return_counts=True)
        c0 = dom[cnt.argmax()].astype(int)
        dist = np.sqrt(((img - c0[None, None, :]) ** 2).sum(-1))
        fill = ink & (dist <= 10)
    if fill.sum() < 100:
        raise SystemExit("no colored or gray fill found — outline-only GIF?")

# --- segment inner polygons: recursive per-component erosion ----------------
# A merged blob (fills touching through a broken outline) needs more erosion
# than a lone shape can survive, so erode each oversized component separately.
expected = fill.sum() / args.n

def split(mask, depth=0):
    lab, k = ndimage.label(mask)
    out = []
    for i in range(1, k + 1):
        c = lab == i
        sz = c.sum()
        if sz < 0.35 * expected:
            continue                                   # antialiasing sliver
        if sz > 1.45 * expected and depth < 8:
            out.extend(split(ndimage.binary_erosion(c), depth + 1))
        else:
            out.append(c)
    return out

cores = split(fill)
if args.debug:
    print(f"split -> {len(cores)} cores (expected {args.n}, "
          f"~{expected:.0f} px each)")
# shapes touching edge-to-edge can survive erosion as one blob: bisect the
# largest cores by 2-means until the count matches
from scipy.cluster.vq import kmeans2
while len(cores) < args.n:
    big = max(range(len(cores)), key=lambda i: cores[i].sum())
    pts_b = np.column_stack(np.nonzero(cores[big])).astype(float)
    _, lab_b = kmeans2(pts_b, 2, minit="++", seed=1)
    if len(set(lab_b)) < 2:
        raise SystemExit("k-means bisection failed")
    for half in (0, 1):
        m = np.zeros_like(fill)
        p = pts_b[lab_b == half].astype(int)
        m[p[:, 0], p[:, 1]] = True
        cores.append(m)
    del cores[big]
    if args.debug:
        print(f"  bisected an oversized core -> {len(cores)}")
# over-segmentation (a shape's fill split by an interior artifact): merge the
# pair with closest centroids — split halves sit ~half a side apart, well
# under the spacing of genuinely distinct shapes
while len(cores) > args.n:
    cents = np.array([np.column_stack(np.nonzero(c)).mean(0) for c in cores])
    d = np.linalg.norm(cents[:, None] - cents[None, :], axis=-1)
    d[np.diag_indices(len(cores))] = 1e9
    i, j = np.unravel_index(d.argmin(), d.shape)
    cores[i] = cores[i] | cores[j]
    del cores[j]
    if args.debug:
        print(f"  merged split cores -> {len(cores)}")
if len(cores) != args.n:
    raise SystemExit(f"segmentation found {len(cores)} shapes, need {args.n}")

# reassign every fill pixel to its nearest core so centroids/orientations use
# full-resolution shapes, not eroded remnants
from scipy.spatial import cKDTree
core_pts = np.vstack([np.column_stack(np.nonzero(c)) for c in cores])
core_lab = np.concatenate([np.full(int(c.sum()), i) for i, c in enumerate(cores)])
fill_pts = np.column_stack(np.nonzero(fill))
_, idx = cKDTree(core_pts).query(fill_pts, k=1)
assign = core_lab[idx]
comps = []
for i in range(args.n):
    m = np.zeros_like(fill)
    p = fill_pts[assign == i]
    m[p[:, 0], p[:, 1]] = True
    comps.append(m)

# --- per-component center + orientation (math coords: y up) -----------------
def center_angle(c):
    rows, cols = np.nonzero(c)
    x, y = cols.astype(float), (H - 1 - rows).astype(float)
    cx, cy = x.mean(), y.mean()
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    ph = np.sum(r ** 3 * np.exp(1j * args.nsi * np.arctan2(dy, dx)))
    return cx, cy, np.angle(ph) / args.nsi

cxy_ang = np.array([center_angle(c) for c in comps])

# --- container corners from convex hull of all ink --------------------------
rows, cols = np.nonzero(ink)
pts = np.column_stack([cols.astype(float), (H - 1 - rows).astype(float)])
hull_pts = pts[ConvexHull(pts).vertices]
# greedy farthest-point pick of nsc corners
corners = [hull_pts[hull_pts[:, 0].argmin()]]
for _ in range(args.nsc - 1):
    d = np.min([np.hypot(*(hull_pts - c).T) for c in corners], axis=0)
    corners.append(hull_pts[d.argmax()])
corners = np.array(corners)
cc = corners.mean(0)
corners = corners[np.argsort(np.arctan2(*(corners - cc).T[::-1]))]  # CCW

# --- similarity transform ----------------------------------------------------
# Scale comes from the shapes themselves (median fill area of a unit-circumradius
# nsi-gon is (nsi/2)sin(2pi/nsi) px^2/unit^2) — far more robust than hull corners.
# The container's center/rotation is then fit directly by minimizing containment
# violation of the reconstructed vertices, which is the quantity we care about.
S_claim = args.s_claim * np.sin(np.pi / args.nsi) / np.sin(np.pi / args.nsc)
areas = np.array([c.sum() for c in comps], float)
k_px = np.sqrt(np.median(areas) / (args.nsi / 2 * np.sin(2 * np.pi / args.nsi)))

# rotation init from hull corners (any cyclic assignment; coarse is fine)
det = corners[:, 0] + 1j * corners[:, 1]
cc_ink = det.mean()
phi0 = np.median(np.mod(np.angle(det - cc_ink), 2 * np.pi / args.nsc))

z_img = cxy_ang[:, 0] + 1j * cxy_ang[:, 1]

# objective = the packer's own penalty (overlap + containment): overlap pushes
# the scale up, containment pushes it down, so (cx, cy, phi, k) is well-posed
import validate_packing as vp
geom = vp.build_geometry(args.nsi, args.nsc)
pen_fn = vp.make_penalty(geom, args.n, args.nsi, args.nsc)
S_fit = S_claim * (1 + args.inflate)

def to_values(p):
    cx, cy, phi, logk = p
    z = (z_img - (cx + 1j * cy)) / (k_px * np.exp(logk)) * np.exp(-1j * phi)
    a = cxy_ang[:, 2] - phi
    return np.column_stack([z.real, z.imag, a]).ravel()

def objective(p):
    return pen_fn(to_values(p), S_fit)

from scipy.optimize import minimize
# rotating the plane by phi changes the container's look with period 2pi/nsc
# but the shapes' with period 2pi/nsi — the multistart must span the JOINT
# period 2pi/gcd, else the fit can alias into a rotation that flips every
# shape's orientation (e.g. diamonds <-> axis-aligned squares in an octagon)
best = None
period = 2 * np.pi / np.gcd(args.nsi, args.nsc)
n_starts = 6 * args.nsc // int(np.gcd(args.nsi, args.nsc))
for dphi in np.linspace(0, period, n_starts, endpoint=False):
    for lk0 in (-0.02, 0.03, 0.08):
        r = minimize(objective, [cc_ink.real, cc_ink.imag, phi0 + dphi, lk0],
                     method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-14, "maxiter": 6000})
        if best is None or r.fun < best.fun:
            best = r
cx, cy, phi, logk = best.x
if args.debug:
    print(f"container fit: penalty {best.fun:.3e}, scale {k_px * np.exp(logk):.3f} "
          f"px/unit (area est {k_px:.3f}), rot {np.degrees(phi):.2f} deg")

vals = to_values(best.x).reshape(args.n, 3)

# --- subpixel per-shape refinement (pure pixel space) ------------------------
# Fit each shape's boundary to its own mask with a FREE per-shape size: the
# drawn fill is the true polygon shrunk by ~half the outline stroke, a shared
# bias that would poison a fixed-size fit. With size free, the recovered
# center/angle are unbiased and ~0.1 px accurate; sizes are then discarded.
cx_, cy_, phi_, logk_ = best.x
k_fit = k_px * np.exp(logk_)
edge_dirs = 2 * np.pi * np.arange(args.nsi) / args.nsi + np.pi / args.nsi
apo_in = np.cos(np.pi / args.nsi)

p_img = vals[:, 0] * 0j                                   # per-shape pixel pose
p_img = (vals[:, 0] + 1j * vals[:, 1]) * k_fit * np.exp(1j * phi_) + (cx_ + 1j * cy_)
th_img = vals[:, 2] + phi_

def to_values2(p):
    cx, cy, phi, logk = p
    z = (p_img - (cx + 1j * cy)) / (k_px * np.exp(logk)) * np.exp(-1j * phi)
    return np.column_stack([z.real, z.imag, th_img - phi]).ravel()

# two passes: shape fits -> global refit -> shape fits again (a systematic
# residual from a misfit global transform vanishes on the second pass)
r2 = best
for fit_pass in range(2):
    for i, c in enumerate(comps):
        ring = ndimage.binary_dilation(c, iterations=3)
        tgt_pix = ring & (fill | ~ink)                    # exclude dark outline
        rows_, cols_ = np.nonzero(tgt_pix)
        px = cols_.astype(float) + 1j * (H - 1 - rows_).astype(float)
        y_tgt = c[rows_, cols_].astype(float)             # 1 own fill, 0 elsewhere

        def make_loss(da0):
            def loss(p):
                ctr = p_img[i] + p[0] + 1j * p[1]
                normals = np.exp(1j * (edge_dirs + th_img[i] + da0 + p[2]))
                proj = np.real((px - ctr)[:, None] * np.conj(normals)[None, :])
                d = proj.max(1) - p[3]                    # p[3]: apothem in px
                pr = 1.0 / (1.0 + np.exp(d / 0.7))
                return np.sum((pr - y_tgt) ** 2)
            return loss

        # the phase-based angle init can alias by a sub-symmetry rotation on
        # antialiased shapes — multistart the boundary fit over the offsets
        # and let the pixels decide
        r, da_best = None, 0.0
        for da0 in np.linspace(0, 2 * np.pi / args.nsi, 4, endpoint=False):
            rr = minimize(make_loss(da0), [0.0, 0.0, 0.0, apo_in * k_fit],
                          method="Nelder-Mead",
                          options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 1200})
            if r is None or rr.fun < r.fun:
                r, da_best = rr, da0

        limit = 4.0 if fit_pass == 0 else 1.5
        # reject divergent fits (clipped/odd components); keep the coarse pose
        if np.hypot(r.x[0], r.x[1]) < limit and abs(r.x[2]) < 0.09 \
                and abs(r.x[3] / (apo_in * k_fit) - 1) < 0.15:
            p_img[i] += r.x[0] + 1j * r.x[1]
            th_img[i] += da_best + r.x[2]
        elif args.debug:
            print(f"  shape {i}: fit rejected pass {fit_pass} "
                  f"(moved {np.hypot(r.x[0], r.x[1]):.2f} px, "
                  f"rot {np.degrees(da_best + r.x[2]):.2f} deg)")

    r2 = minimize(lambda p: pen_fn(to_values2(p), S_fit), r2.x,
                  method="Nelder-Mead",
                  options={"xatol": 1e-9, "fatol": 1e-16, "maxiter": 8000})
    cx_, cy_, phi_, logk_ = r2.x
    k_fit = k_px * np.exp(logk_)
    if args.debug:
        print(f"pass {fit_pass}: penalty {r2.fun:.3e}, scale {k_fit:.3f} px/unit, "
              f"rot {np.degrees(phi_):.2f} deg")

vals = to_values2(r2.x).reshape(args.n, 3)
z = vals[:, 0] + 1j * vals[:, 1]
ang = np.mod(vals[:, 2], 2 * np.pi / args.nsi)

S_out = S_claim * (1 + args.inflate)
ratio = np.sin(np.pi / args.nsc) / np.sin(np.pi / args.nsi)
out = args.out or args.gif.rsplit(".", 1)[0] + "_recon.json"
with open(out, "w") as f:
    json.dump({
        "inner_polygons": args.n, "inner_sides": args.nsi, "container_sides": args.nsc,
        "container_circumradius": float(S_out),
        "side_length": float(S_out) * ratio,
        "reconstructed_from": args.gif,
        "placements": [{"x": float(zi.real), "y": float(zi.imag), "angle": float(a)}
                       for zi, a in zip(z, ang)],
    }, f, indent=2)
print(f"{args.n} shapes reconstructed -> {out}")
print(f"S = {S_out:.6f} (claimed {S_claim:.6f} + {args.inflate:.1%} slack), "
      f"side ratio = {S_out * ratio:.6f}")
