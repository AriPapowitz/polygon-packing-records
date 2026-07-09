"""Validate, polish, and certify packings produced by polygon_packer.py.

polygon_packer.py accepts a packing once its penalty (sum of squared overlaps
and squared container violations) drops below ~1e-8, which still tolerates
overlaps on the order of 1e-4 -- far too loose to claim a packing record.

This script closes that gap:
  1. VALIDATE: recompute exact separation margins for every polygon pair
     (Separating Axis Theorem over both polygons' edge normals -- exact for
     convex polygons) and exact containment margins for every vertex.
  2. POLISH (--polish): re-minimize the penalty at fixed container size with
     machine-precision tolerance, driving residual overlap to ~1e-14.
  3. SQUEEZE (--squeeze): after polishing, keep shrinking the container in
     backtracking steps as long as a machine-precision-valid packing survives.
  4. CERTIFY: convert any residual violation into a rigorous bound by
     shrinking the inner polygons about their centers (fixes overlap) and
     dilating the container (fixes poking), then report the certified
     container side length -- a size at which the packing is PROVABLY valid.

Usage:
    python validate_packing.py 8_3_in_6.json
    python validate_packing.py 8_3_in_6.json --polish
    python validate_packing.py 8_3_in_6.json --polish --squeeze
"""

import argparse
import json
import os

import numpy as np
from numba import njit
from scipy.optimize import minimize


# ---------------------------------------------------------------- geometry

def build_geometry(nsi, nsc):
    inner_angles = np.linspace(0, 2 * np.pi, nsi, endpoint=False)
    container_angles = np.linspace(0, 2 * np.pi, nsc, endpoint=False)
    return {
        "inner_vertices": np.column_stack((np.cos(inner_angles), np.sin(inner_angles))),
        "inner_normals": np.column_stack((np.cos(inner_angles + np.pi / nsi), np.sin(inner_angles + np.pi / nsi))),
        "container_normals": np.column_stack((np.cos(container_angles + np.pi / nsc), np.sin(container_angles + np.pi / nsc))),
        "inner_apothem": np.cos(np.pi / nsi),
        "container_apothem": np.cos(np.pi / nsc),
    }


def transform_all(values, inner_vertices, inner_normals):
    """Return (N,nsi,2) world-space vertices and edge normals for all polygons."""
    n = len(values) // 3
    xy = values.reshape(n, 3)[:, :2]
    a = values.reshape(n, 3)[:, 2]
    cos_a, sin_a = np.cos(a), np.sin(a)
    # rotation applied to each unit vertex, then translated
    rot = np.stack([np.stack([cos_a, -sin_a], -1), np.stack([sin_a, cos_a], -1)], -2)  # (N,2,2)
    verts = np.einsum("nij,vj->nvi", rot, inner_vertices) + xy[:, None, :]
    normals = np.einsum("nij,vj->nvi", rot, inner_normals)
    return verts, normals


# ---------------------------------------------------------------- exact checks

def pair_separation(verts_i, verts_j, normals_i, normals_j):
    """Exact SAT margin for one pair: > 0 means disjoint by that gap,
    < 0 means penetration of that depth (minimum translation along an edge normal)."""
    axes = np.vstack((normals_i, normals_j))            # (2*nsi, 2)
    proj_i = verts_i @ axes.T                           # (nsi, axes)
    proj_j = verts_j @ axes.T
    overlap = (np.minimum(proj_i.max(0), proj_j.max(0))
               - np.maximum(proj_i.min(0), proj_j.min(0)))
    return (-overlap).max()


def exact_margins(values, S, geom):
    """Worst pair-separation margin and worst containment margin (both >= 0 for a valid packing)."""
    verts, normals = transform_all(values, geom["inner_vertices"], geom["inner_normals"])
    n = verts.shape[0]

    worst_pair = np.inf
    touching = 0
    for i in range(n):
        for j in range(i + 1, n):
            sep = pair_separation(verts[i], verts[j], normals[i], normals[j])
            if abs(sep) < 1e-7:
                touching += 1
            if sep < worst_pair:
                worst_pair = sep

    limit = S * geom["container_apothem"]
    projections = verts.reshape(-1, 2) @ geom["container_normals"].T
    worst_containment = (limit - projections).min()

    centers_inside = bool((values.reshape(n, 3)[:, :2] @ geom["container_normals"].T <= limit).all())
    return worst_pair, worst_containment, touching, centers_inside


# ---------------------------------------------------------------- penalty (matches polygon_packer.py)

@njit(cache=True)
def _penalty(values, S, inner_vertices, inner_normals, container_normals, container_apothem, n, nsi, nsc):
    penalty = 0.0
    polygon_array = np.zeros((n, nsi, 2))
    vector_array = np.zeros((n, nsi, 2))
    limit = container_apothem * S
    for i in range(n):
        posx, posy, rot = values[i * 3], values[i * 3 + 1], values[i * 3 + 2]
        sina, cosa = np.sin(rot), np.cos(rot)
        for v in range(nsi):
            vx, vy = inner_vertices[v, 0], inner_vertices[v, 1]
            polygon_array[i, v, 0] = posx + vx * cosa - vy * sina
            polygon_array[i, v, 1] = posy + vx * sina + vy * cosa
            nx, ny = inner_normals[v, 0], inner_normals[v, 1]
            vector_array[i, v, 0] = nx * cosa - ny * sina
            vector_array[i, v, 1] = nx * sina + ny * cosa
        for v in range(nsi):
            for c in range(nsc):
                distance = (polygon_array[i, v, 0] * container_normals[c, 0]
                            + polygon_array[i, v, 1] * container_normals[c, 1])
                if distance > limit:
                    diff = distance - limit
                    penalty += diff * diff

    for i in range(n):
        for j in range(i + 1, n):
            collision = True
            min_overlap = 1e20
            for vec in range(nsi * 2):
                if vec < nsi:
                    x_axis, y_axis = vector_array[i][vec, 0], vector_array[i][vec, 1]
                else:
                    x_axis, y_axis = vector_array[j][vec - nsi, 0], vector_array[j][vec - nsi, 1]
                min_1, max_1 = 1e20, -1e20
                min_2, max_2 = 1e20, -1e20
                for vert in range(nsi):
                    dotp = polygon_array[i][vert, 0] * x_axis + polygon_array[i][vert, 1] * y_axis
                    if dotp < min_1: min_1 = dotp
                    if dotp > max_1: max_1 = dotp
                    dotp = polygon_array[j][vert, 0] * x_axis + polygon_array[j][vert, 1] * y_axis
                    if dotp < min_2: min_2 = dotp
                    if dotp > max_2: max_2 = dotp
                overlap = min(max_1, max_2) - max(min_1, min_2)
                if overlap <= 0:
                    collision = False
                    break
                if overlap < min_overlap:
                    min_overlap = overlap
            if collision:
                penalty += min_overlap * min_overlap
    return penalty


def make_penalty(geom, n, nsi, nsc):
    iv, inn = geom["inner_vertices"], geom["inner_normals"]
    cn, ca = geom["container_normals"], geom["container_apothem"]

    def f(values, S):
        return _penalty(values, S, iv, inn, cn, ca, n, nsi, nsc)
    return f


# ---------------------------------------------------------------- polish / squeeze

POLISH_TARGET = 1e-26  # residual penalty ~ (1e-13)^2 per contact


def polish(values, S, penalty_fn, max_rounds=8):
    """Re-minimize the penalty at fixed S until it stops improving."""
    best = values.copy()
    best_pen = penalty_fn(best, S)
    for _ in range(max_rounds):
        res = minimize(penalty_fn, best, args=(S,), method="L-BFGS-B",
                       tol=0, options={"maxiter": 20000, "ftol": 0, "gtol": 1e-16})
        if res.fun < best_pen:
            best, best_pen = res.x.copy(), res.fun
        else:
            break
        if best_pen < POLISH_TARGET:
            break
    return best, best_pen


def squeeze(values, S, penalty_fn, initial_step=1e-5, min_step=1e-11):
    """Shrink S in backtracking multiplicative steps while a machine-valid packing survives."""
    step = initial_step
    values, pen = polish(values, S, penalty_fn)
    while step >= min_step:
        S_try = S * (1 - step)
        cand, cand_pen = polish(values * (1 - step), S_try, penalty_fn)
        if cand_pen < POLISH_TARGET:
            values, S, pen = cand, S_try, cand_pen
        else:
            step /= 4
    return values, S, pen


# ---------------------------------------------------------------- certification

def certify(values, S, geom, nsi, nsc):
    """Rigorous size: shrink inner polygons to erase overlap, dilate container to erase poking.
    Returns the certified 'side length' in the Packing Center's convention."""
    worst_pair, worst_containment, touching, centers_inside = exact_margins(values, S, geom)
    d = max(0.0, -worst_pair)           # deepest pairwise penetration
    p = max(0.0, -worst_containment)    # deepest container violation
    k = 1 - d / (2 * geom["inner_apothem"])          # shrink factor for inner polygons
    S_cert = (S + p / geom["container_apothem"]) / k  # rescale so inner polygons are unit again
    ratio = S * np.sin(np.pi / nsc) / np.sin(np.pi / nsi)
    ratio_cert = S_cert * np.sin(np.pi / nsc) / np.sin(np.pi / nsi)
    return {
        "worst_pair_separation": worst_pair,
        "worst_containment_margin": worst_containment,
        "touching_pairs": touching,
        "centers_inside": centers_inside,
        "raw_side_length": ratio,
        "certified_side_length": ratio_cert,
        "certified_penalty_inflation": ratio_cert - ratio,
    }


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("solution", help="Solution .json produced by polygon_packer.py")
    parser.add_argument("--polish", action="store_true", help="Re-minimize at fixed size to machine precision")
    parser.add_argument("--squeeze", action="store_true", help="After polishing, shrink the container as far as it will go")
    args = parser.parse_args()

    with open(args.solution) as f:
        sol = json.load(f)

    n, nsi, nsc = sol["inner_polygons"], sol["inner_sides"], sol["container_sides"]
    S = float(sol["container_circumradius"])
    values = np.array([[p["x"], p["y"], p["angle"]] for p in sol["placements"]]).ravel()

    geom = build_geometry(nsi, nsc)
    penalty_fn = make_penalty(geom, n, nsi, nsc)

    print(f"Loaded {n} {nsi}-gons in a {nsc}-gon  (S = {S!r})")
    print(f"Initial penalty: {penalty_fn(values, S):.3e}")

    if args.squeeze:
        values, S, pen = squeeze(values, S, penalty_fn)
        print(f"After squeeze:  S = {S!r}   penalty = {pen:.3e}")
    elif args.polish:
        values, pen = polish(values, S, penalty_fn)
        print(f"After polish:   penalty = {pen:.3e}")

    report = certify(values, S, geom, nsi, nsc)
    print()
    print("=== Validity report ===")
    print(f"Worst pair separation:    {report['worst_pair_separation']:+.3e}  (>= 0 is valid)")
    print(f"Worst containment margin: {report['worst_containment_margin']:+.3e}  (>= 0 is valid)")
    print(f"Touching pairs (|sep| < 1e-7): {report['touching_pairs']}")
    if not report["centers_inside"]:
        print("WARNING: some polygon centers are outside the container; certification is unreliable.")
    print()
    print("=== Size (container side / inner side) ===")
    print(f"Raw:       {report['raw_side_length']:.15f}")
    print(f"Certified: {report['certified_side_length']:.15f}")
    print(f"Certification cost: {report['certified_penalty_inflation']:.3e}")

    if args.polish or args.squeeze:
        stem = os.path.splitext(args.solution)[0] + "_polished"
        out = stem + ".json"
        file_i = 1
        while os.path.exists(out):
            out = f"{stem}_({file_i}).json"
            file_i += 1
        with open(out, "w") as f:
            json.dump({
                "inner_polygons": n,
                "inner_sides": nsi,
                "container_sides": nsc,
                "container_circumradius": S,
                "side_length": report["raw_side_length"],
                "certified_side_length": report["certified_side_length"],
                "placements": [
                    {"x": values[i * 3], "y": values[i * 3 + 1], "angle": values[i * 3 + 2]}
                    for i in range(n)
                ],
            }, f, indent=2)
        print(f"\nSaved polished solution to {out}")


if __name__ == "__main__":
    main()
