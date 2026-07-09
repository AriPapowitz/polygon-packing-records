"""Scan removal variants of the 24-triangle lattice for squeezable configurations.

For each removal set, start from the exact lattice-minus-holes configuration at
s = 2.000 and run a JAX-gradient squeeze (shrink container with backtracking,
re-minimizing the packing penalty after each step). Configurations whose holes
let the packing rearrange will squeeze below 2.000 -> record candidates for
n = 22/23 triangles in a hexagon (both currently trivial 2.000 entries).

Each removal set is tried from the unperturbed lattice plus a few jittered
starts (pure descent from a rigid tiling often needs a nudge to rearrange).

Usage:
    python scan_removals.py --singles 0-23            # n=23 scan
    python scan_removals.py --pairs "0,1;0,5;3,17"    # explicit n=22 configs
    python scan_removals.py --auto-pairs 0 8          # auto-generate pair list, take slice [0:8]
Writes best result per config to stdout and the best packing JSONs to disk.
"""

import argparse
import json

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

from build_lattice import lattice_placements, S as LATTICE_S
from validate_packing import build_geometry, exact_margins

NSI, NSC = 3, 6
POLISH_TARGET = 1e-26


def make_jax_penalty(n):
    inner_angles = np.linspace(0, 2 * np.pi, NSI, endpoint=False)
    unit_vertices = jnp.array(np.column_stack((np.cos(inner_angles), np.sin(inner_angles))))
    unit_normals = jnp.array(np.column_stack((np.cos(inner_angles + np.pi / NSI), np.sin(inner_angles + np.pi / NSI))))
    cont_angles = np.linspace(0, 2 * np.pi, NSC, endpoint=False)
    cont_normals = jnp.array(np.column_stack((np.cos(cont_angles + np.pi / NSC), np.sin(cont_angles + np.pi / NSC))))
    cont_apothem = float(np.cos(np.pi / NSC))
    pi_, pj_ = np.triu_indices(n, k=1)

    @jax.jit
    def penalty(values, S):
        v = values.reshape(n, 3)
        xy, a = v[:, :2], v[:, 2]
        cos_a, sin_a = jnp.cos(a), jnp.sin(a)
        rot = jnp.stack([jnp.stack([cos_a, -sin_a], -1), jnp.stack([sin_a, cos_a], -1)], -2)
        verts = jnp.einsum("nij,vj->nvi", rot, unit_vertices) + xy[:, None, :]
        normals = jnp.einsum("nij,vj->nvi", rot, unit_normals)
        proj_wall = jnp.einsum("nvi,ci->nvc", verts, cont_normals)
        pen = jnp.sum(jnp.maximum(proj_wall - cont_apothem * S, 0.0) ** 2)
        axes = jnp.concatenate([normals[pi_], normals[pj_]], axis=1)
        proj_i = jnp.einsum("pvi,pai->pav", verts[pi_], axes)
        proj_j = jnp.einsum("pvi,pai->pav", verts[pj_], axes)
        overlap = (jnp.minimum(proj_i.max(-1), proj_j.max(-1))
                   - jnp.maximum(proj_i.min(-1), proj_j.min(-1)))
        pen += jnp.sum(jnp.maximum(overlap.min(-1), 0.0) ** 2)
        return pen

    vg = jax.jit(jax.value_and_grad(penalty))

    def scipy_obj(x, S):
        val, grad = vg(x, S)
        return float(val), np.asarray(grad)
    return scipy_obj


def polish(obj, x, S, rounds=4):
    best, best_pen = x, obj(x, S)[0]
    for _ in range(rounds):
        res = minimize(obj, best, args=(S,), method="L-BFGS-B", jac=True,
                       tol=0, options={"maxiter": 20000, "ftol": 0, "gtol": 1e-16})
        if res.fun >= best_pen:
            break
        best, best_pen = res.x.copy(), res.fun
        if best_pen < POLISH_TARGET:
            break
    return best, best_pen


def squeeze(obj, x, S, initial_step=3e-4, min_step=1e-9):
    x, pen = polish(obj, x, S)
    step = initial_step
    while step >= min_step:
        cand, cand_pen = polish(obj, x * (1 - step), S * (1 - step))
        if cand_pen < POLISH_TARGET:
            x, S, pen = cand, S * (1 - step), cand_pen
        else:
            step /= 4
    return x, S


def scan_config(remove, jitter_seeds=(None, 1, 2, "kick3", "kick4", "kick5")):
    """Try one removal set from the lattice; return (best_ratio, best_x, best_S).

    Seeds: None = exact lattice start; int = Gaussian jitter; "kickN" = jitter plus
    a few triangles respun to fully random angles (local descent cannot rotate a
    triangle 30 deg on its own, and real rearrangements often need that)."""
    placements = lattice_placements()
    keep = np.array([p for i, p in enumerate(placements) if i not in set(remove)])
    n = len(keep)
    obj = make_jax_penalty(n)
    geom = build_geometry(NSI, NSC)

    best_ratio, best_x, best_S = np.inf, None, None
    for js in jitter_seeds:
        x0 = keep.ravel().copy()
        if isinstance(js, int):
            rng = np.random.RandomState(1000 + js)
            x0 = x0 + rng.normal(0, 0.02 * js, x0.shape)
        elif isinstance(js, str) and js.startswith("kick"):
            rng = np.random.RandomState(2000 + int(js[4:]))
            x0 = x0 + rng.normal(0, 0.02, x0.shape)
            respin = rng.choice(n, size=max(2, n // 5), replace=False)
            x0[respin * 3 + 2] = rng.uniform(0, 2 * np.pi, len(respin))
        x, S = squeeze(obj, x0, LATTICE_S)
        # exact validity check before trusting the ratio
        wp, wc, _, _ = exact_margins(x, S, geom)
        if wp < -1e-9 or wc < -1e-9:
            continue
        ratio = S * np.sin(np.pi / NSC) / np.sin(np.pi / NSI)
        if ratio < best_ratio:
            best_ratio, best_x, best_S = ratio, x, S
    return best_ratio, best_x, best_S


def auto_pairs():
    """Candidate pairs: all edge/vertex-sharing neighbors + a spread of far pairs."""
    placements = lattice_placements()
    cents = np.array([[x, y] for (x, y, _) in placements])
    pairs = []
    for i in range(24):
        for j in range(i + 1, 24):
            pairs.append((np.linalg.norm(cents[i] - cents[j]), i, j))
    pairs.sort()
    near = [(i, j) for d, i, j in pairs if d <= 1.85]          # edge + vertex neighbors
    far = [(i, j) for d, i, j in pairs[len(near)::12]]          # sparse sample of the rest
    return near + far


def auto_triples():
    """Candidate triples: 3-fold rotationally symmetric orbits (likely the n=21
    record's structure) + mutually-near clusters."""
    placements = lattice_placements()
    arr = np.array(placements)
    cents = arr[:, :2]

    # orbits under 120-degree rotation about the center
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    rot_c = cents @ np.array([[c, s], [-s, c]])  # rotate each centroid by +120 deg
    mapping = [int(np.argmin(np.linalg.norm(cents - rc, axis=1))) for rc in rot_c]
    orbits = set()
    for i in range(24):
        orbit = tuple(sorted({i, mapping[i], mapping[mapping[i]]}))
        if len(orbit) == 3:
            orbits.add(orbit)

    # mutually-near clusters (all pairwise centroid distances small)
    clusters = set()
    for i in range(24):
        for j in range(i + 1, 24):
            if np.linalg.norm(cents[i] - cents[j]) > 1.85:
                continue
            for k in range(j + 1, 24):
                if (np.linalg.norm(cents[i] - cents[k]) <= 1.85
                        and np.linalg.norm(cents[j] - cents[k]) <= 1.85):
                    clusters.add((i, j, k))
    clusters = sorted(clusters)[::max(1, len(clusters) // 16)][:16]
    return sorted(orbits) + list(clusters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles", type=str, default=None, help="e.g. 0-23")
    ap.add_argument("--pairs", type=str, default=None, help="e.g. '0,1;0,5;3,17'")
    ap.add_argument("--auto-pairs", type=int, nargs=2, default=None,
                    metavar=("START", "COUNT"), help="slice of auto-generated pair list")
    ap.add_argument("--auto-triples", type=int, nargs=2, default=None,
                    metavar=("START", "COUNT"), help="slice of auto-generated triple list (n=21 control)")
    args = ap.parse_args()

    configs = []
    if args.singles:
        lo, hi = map(int, args.singles.split("-"))
        configs = [(i,) for i in range(lo, hi + 1)]
    elif args.pairs:
        configs = [tuple(map(int, p.split(","))) for p in args.pairs.split(";")]
    elif args.auto_pairs:
        start, count = args.auto_pairs
        configs = auto_pairs()[start:start + count]
    elif args.auto_triples:
        start, count = args.auto_triples
        configs = auto_triples()[start:start + count]

    overall_best = np.inf
    for remove in configs:
        ratio, x, S = scan_config(remove)
        tag = "-".join(map(str, remove))
        print(f"remove {tag}: ratio = {ratio:.9f}" + ("  <<< SUB-2!" if ratio < 2.0 - 1e-6 else ""), flush=True)
        if ratio < min(2.0 - 1e-6, overall_best):
            overall_best = ratio
            n = 24 - len(remove)
            with open(f"{n}_3_in_6_scan_rm{tag}.json", "w") as f:
                json.dump({
                    "inner_polygons": n, "inner_sides": NSI, "container_sides": NSC,
                    "container_circumradius": float(S), "side_length": float(ratio),
                    "placements": [{"x": float(x[i * 3]), "y": float(x[i * 3 + 1]),
                                    "angle": float(x[i * 3 + 2])} for i in range(n)],
                }, f, indent=2)
    print(f"BEST: {overall_best}")


if __name__ == "__main__":
    main()
