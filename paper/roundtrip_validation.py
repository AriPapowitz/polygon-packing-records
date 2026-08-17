"""Round-trip validation of the reconstruction (harvest) instrument.

For every solution in paper/solutions/: render it in the site's image style at
the site's typical pixel scale (~240 px, gray fills, thin dark outlines, white
background), reconstruct coordinates from that image with reconstruct_gif.py,
and measure
  (1) geometric error: per-shape center error in pixels after the optimal
      container-symmetry alignment (all nsc rotations x reflection, matched by
      linear-sum assignment),
  (2) basin recovery: polish + squeeze the reconstruction in float64 and check
      whether it re-converges to the source packing's side length (<1e-9).
(2) is the assumption behind reading harvest margins as the incumbent's own
convergence gap; this experiment measures how often it holds on ground truth.

Run:  polygon-packer/.venv/Scripts/python paper/roundtrip_validation.py
Output: paper/roundtrip.csv (+ console log)
"""
import csv, json, math, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
PP = ROOT / "polygon-packer"
SOL = ROOT / "paper" / "solutions"
PY = str(PP / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(PP))
from pack_core import Engine  # noqa: E402

TARGET_PX = 240
SS = 8  # supersampling

def render_site_style(sol, out_png):
    """Gray fills, black outlines, white bg, container fit to ~TARGET_PX."""
    n, nsi, nsc = sol["inner_polygons"], sol["inner_sides"], sol["container_sides"]
    S = sol["container_circumradius"]
    pad = 6
    scale = (TARGET_PX - 2 * pad) / (2 * S)
    W = TARGET_PX * SS
    img = Image.new("RGB", (W, W), "white")
    dr = ImageDraw.Draw(img)
    def to_px(x, y):
        return ((x + S) * scale + pad) * SS, ((S - y) * scale + pad) * SS
    ca = np.linspace(0, 2 * np.pi, nsc, endpoint=False)
    cont = [to_px(S * math.cos(a), S * math.sin(a)) for a in ca]
    dr.polygon(cont, outline="black", width=SS)
    ia = np.linspace(0, 2 * np.pi, nsi, endpoint=False)
    for p in sol["placements"]:
        c, s = math.cos(p["angle"]), math.sin(p["angle"])
        pts = [to_px(p["x"] + math.cos(a) * c - math.sin(a) * s,
                     p["y"] + math.cos(a) * s + math.sin(a) * c) for a in ia]
        dr.polygon(pts, fill=(204, 204, 204), outline="black", width=SS)
    img = img.resize((TARGET_PX, TARGET_PX), Image.LANCZOS)
    img.save(out_png)
    return scale  # px per model unit

def aligned_error(true_xy, true_a, rec_xy, rec_a, nsi, nsc):
    """Best center/angle error over the container's symmetry group."""
    best = None
    for refl in (1, -1):
        for k in range(nsc):
            th = 2 * math.pi * k / nsc
            R = np.array([[math.cos(th), -math.sin(th)],
                          [math.sin(th), math.cos(th)]])
            xy = rec_xy.copy()
            if refl == -1:
                xy = xy * np.array([1.0, -1.0])
            xy = xy @ R.T
            D = np.linalg.norm(true_xy[:, None, :] - xy[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(D)
            errs = D[ri, ci]
            if best is None or errs.mean() < best[0]:
                da = rec_a[ci] * refl + th - true_a[ri]
                per = 2 * math.pi / nsi
                da = (da + per / 2) % per - per / 2
                best = (errs.mean(), errs.max(), np.abs(da).mean())
    return best

rows = []
t_all = time.time()
for f in sorted(SOL.glob("*.json")):
    cat, ns = f.stem.rsplit("_", 1)
    n = int(ns)
    sol = json.load(open(f, encoding="utf-8"))
    nsi, nsc = sol["inner_sides"], sol["container_sides"]
    s_true = sol["side_length"]
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "r.png"
        scale = render_site_style(sol, png)
        rec_path = Path(td) / "rec.json"
        r = subprocess.run(
            [PY, str(PP / "reconstruct_gif.py"), str(png), str(n), str(nsi),
             str(nsc), f"{s_true:.6f}", "--out", str(rec_path)],
            capture_output=True, text=True, timeout=300)
        if not rec_path.exists():
            rows.append([cat, n, f"{scale:.2f}", "", "", "", "", "",
                         "RECON-FAILED"])
            print(f"{cat} n={n}: reconstruction FAILED "
                  f"({r.stdout.strip()[-80:]} {r.stderr.strip()[-80:]})", flush=True)
            continue
        rec = json.load(open(rec_path, encoding="utf-8"))
        true_xy = np.array([[p["x"], p["y"]] for p in sol["placements"]])
        true_a = np.array([p["angle"] for p in sol["placements"]])
        rec_xy = np.array([[p["x"], p["y"]] for p in rec["placements"]])
        rec_a = np.array([p["angle"] for p in rec["placements"]])
        mean_e, max_e, ang_e = aligned_error(true_xy, true_a, rec_xy, rec_a, nsi, nsc)
        # basin recovery: polish -> grow -> squeeze the reconstruction (float64)
        t0 = time.time()
        eng = Engine(n, nsi, nsc)
        vals = np.array([[p["x"], p["y"], p["angle"]]
                         for p in rec["placements"]]).ravel()[None, :]
        S0 = np.array([float(rec["container_circumradius"])])
        S_r, x_r, valid = eng.refine64(vals, S0, iters=300, grow_rounds=40,
                                       squeeze_rounds=150, squeeze_step=4e-4)
        if not bool(valid[0]):
            rows.append([cat, n, f"{scale:.2f}", f"{mean_e * scale:.3f}",
                         f"{max_e * scale:.3f}", f"{ang_e:.4f}", "", "",
                         "REFINE-INVALID"])
            print(f"{cat} n={n}: err {mean_e*scale:.2f}px mean, refine "
                  f"INVALID [{time.time()-t0:.0f}s]", flush=True)
            continue
        s_rec = float(S_r[0]) * eng.ratio
        d = s_rec - s_true
        same = abs(d) < 1e-9
        verdict = ("SAME-BASIN" if same else
                   ("BELOW-CHECK" if d < 0 else "ABOVE"))
        if verdict == "BELOW-CHECK":
            below = ROOT / "paper" / "roundtrip_below"
            below.mkdir(exist_ok=True)
            xr = np.asarray(x_r)[0].reshape(n, 3)
            json.dump(dict(inner_polygons=n, inner_sides=nsi,
                           container_sides=nsc,
                           container_circumradius=float(S_r[0]),
                           side_length=s_rec,
                           placements=[dict(x=float(a), y=float(b),
                                            angle=float(c)) for a, b, c in xr],
                           method="roundtrip squeeze landed below source"),
                      open(below / f"{cat}_{n:02d}.json", "w"), indent=1)
        rows.append([cat, n, f"{scale:.2f}", f"{mean_e * scale:.3f}",
                     f"{max_e * scale:.3f}", f"{ang_e:.4f}", f"{s_rec:.12f}",
                     f"{d:+.3e}", verdict])
        print(f"{cat} n={n}: err {mean_e*scale:.2f}px mean / {max_e*scale:.2f}px max, "
              f"squeezed {s_rec:.9f} vs {s_true:.9f} ({d:+.1e}) "
              f"{verdict} [{time.time()-t0:.0f}s]", flush=True)

with open(ROOT / "paper" / "roundtrip.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["category", "n", "px_per_unit", "center_err_px_mean",
                "center_err_px_max", "angle_err_rad_mean", "s_after_squeeze",
                "delta_vs_true", "verdict"])
    w.writerows(rows)
ok = sum(1 for r in rows if r[-1] == "SAME-BASIN")
print(f"done in {(time.time()-t_all)/60:.0f} min: {ok}/{len(rows)} same-basin "
      f"-> paper/roundtrip.csv")
