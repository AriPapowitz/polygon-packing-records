"""JAX-accelerated polygon packer prototype.

Same problem and penalty semantics as polygon_packer.py, but:
  - the penalty is a fully vectorized, differentiable JAX function
  - L-BFGS-B receives an EXACT autodiff gradient (value_and_grad) instead of
    estimating it by finite differences (~3N+1 penalty calls per step)
  - float64 enabled; runs on CPU today, on GPU by installing a CUDA jaxlib

Penalty equivalence with the original:
  original: for each colliding pair (all SAT axes overlap > 0), add min_overlap^2
  here:     pen_pair = relu(min over axes of overlap)^2  -- identical, since
            min(overlaps) > 0  <=>  all overlaps > 0
  original: for each vertex poking distance d past a container wall, add d^2
  here:     pen_wall = sum relu(proj - limit)^2          -- identical

Usage:
    python packer_jax.py [n] [nsi] [nsc] [--attempts A] [--tolerance T] [--finalstep F]
"""

import argparse
import json
import os
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import basinhopping, minimize

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("inner_polygons", type=int)
arg_parser.add_argument("inner_sides", type=int)
arg_parser.add_argument("container_sides", type=int)
arg_parser.add_argument("--attempts", type=int, default=100)
arg_parser.add_argument("--tolerance", type=float, default=1e-8)
arg_parser.add_argument("--finalstep", type=float, default=0.0001)
arg_parser.add_argument("--seed0", type=int, default=0, help="First random seed (attempts use seed0..seed0+attempts-1)")
args = arg_parser.parse_args()

N = args.inner_polygons
nsi = args.inner_sides
nsc = args.container_sides
penalty_tolerance = args.tolerance
final_step_size = args.finalstep

inner_angles = np.linspace(0, 2 * np.pi, nsi, endpoint=False)
UNIT_VERTICES = jnp.array(np.column_stack((np.cos(inner_angles), np.sin(inner_angles))))
UNIT_NORMALS = jnp.array(np.column_stack((np.cos(inner_angles + np.pi / nsi), np.sin(inner_angles + np.pi / nsi))))
container_angles = np.linspace(0, 2 * np.pi, nsc, endpoint=False)
CONTAINER_NORMALS = jnp.array(np.column_stack((np.cos(container_angles + np.pi / nsc), np.sin(container_angles + np.pi / nsc))))
CONTAINER_APOTHEM = float(np.cos(np.pi / nsc))
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
# area lower bound on the container circumradius (reduces to sqrt(N) for same shapes)
AREA_LOWER_BOUND_S = float(np.sqrt(N * (nsi / 2 * np.sin(2 * np.pi / nsi))
                                   / (nsc / 2 * np.sin(2 * np.pi / nsc))))


@jax.jit
def penalty(values, S):
    v = values.reshape(N, 3)
    xy, a = v[:, :2], v[:, 2]
    cos_a, sin_a = jnp.cos(a), jnp.sin(a)
    rot = jnp.stack([jnp.stack([cos_a, -sin_a], -1), jnp.stack([sin_a, cos_a], -1)], -2)  # (N,2,2)
    verts = jnp.einsum("nij,vj->nvi", rot, UNIT_VERTICES) + xy[:, None, :]                # (N,nsi,2)
    normals = jnp.einsum("nij,vj->nvi", rot, UNIT_NORMALS)                                # (N,nsi,2)

    # container violation
    limit = CONTAINER_APOTHEM * S
    proj_wall = jnp.einsum("nvi,ci->nvc", verts, CONTAINER_NORMALS)
    pen_wall = jnp.sum(jnp.maximum(proj_wall - limit, 0.0) ** 2)

    # pairwise SAT overlap (axes = edge normals of both polygons of the pair)
    axes = jnp.concatenate([normals[PAIR_I], normals[PAIR_J]], axis=1)   # (P, 2*nsi, 2)
    proj_i = jnp.einsum("pvi,pai->pav", verts[PAIR_I], axes)             # (P, axes, verts)
    proj_j = jnp.einsum("pvi,pai->pav", verts[PAIR_J], axes)
    overlap = (jnp.minimum(proj_i.max(-1), proj_j.max(-1))
               - jnp.maximum(proj_i.min(-1), proj_j.min(-1)))            # (P, axes)
    depth = overlap.min(-1)                                              # (P,) SAT penetration
    pen_pairs = jnp.sum(jnp.maximum(depth, 0.0) ** 2)

    return pen_wall + pen_pairs


_value_and_grad = jax.jit(jax.value_and_grad(penalty))


def scipy_obj(x, S):
    val, grad = _value_and_grad(x, S)
    return float(val), np.asarray(grad)


def repetition(seed):
    """Same annealed-shrink schedule as polygon_packer.repetition, with exact gradients."""
    rng = np.random.RandomState(seed)
    dynamic_S = AREA_LOWER_BOUND_S * (2 + rng.rand() * 2)
    initial_S = dynamic_S
    lowest_S = AREA_LOWER_BOUND_S
    span = initial_S - lowest_S

    if rng.rand() < 0.5:
        x0 = rng.uniform(-dynamic_S / 2, dynamic_S / 2, N * 3)
    else:
        grid = np.linspace(-dynamic_S / 2 * 0.9, dynamic_S / 2 * 0.9, int(np.ceil(np.sqrt(N))))
        xx, yy = np.meshgrid(grid, grid)
        pts = np.column_stack((xx.ravel(), yy.ravel()))[:N]
        x0 = np.zeros(N * 3)
        x0[0::3], x0[1::3] = pts[:, 0], pts[:, 1]
        x0[2::3] = rng.uniform(0, 2 * np.pi, N)

    last_valid_x, last_valid_S = x0.copy(), dynamic_S

    while True:
        res = minimize(scipy_obj, x0, args=(dynamic_S,), method="L-BFGS-B", jac=True, tol=1e-8)
        multiplier = 1 - final_step_size - (dynamic_S - lowest_S) * (0.01 - final_step_size) / span
        if multiplier >= 1:  # safety: S must always shrink, or the loop never terminates
            multiplier = 1 - final_step_size
        if res.fun < penalty_tolerance:
            last_valid_x, last_valid_S = res.x.copy(), dynamic_S
            x0 = res.x * multiplier
            dynamic_S *= multiplier
        else:
            bh = basinhopping(
                scipy_obj, x0,
                minimizer_kwargs={"method": "L-BFGS-B", "jac": True, "args": (dynamic_S,), "tol": 1e-8},
                niter=50, T=0.1, stepsize=0.1,
            )
            if bh.fun < penalty_tolerance:
                last_valid_x, last_valid_S = bh.x.copy(), dynamic_S
                x0 = bh.x * multiplier
                dynamic_S *= multiplier
            else:
                break
    return last_valid_S, last_valid_x


if __name__ == "__main__":
    t0 = time.perf_counter()
    best_S, best_values = float("inf"), None
    for seed in range(args.seed0, args.seed0 + args.attempts):
        s, vals = repetition(seed)
        marker = " *" if s < best_S else ""
        print(f"Attempt {seed}: S = {s:.6f}  ({time.perf_counter() - t0:.1f}s total){marker}", flush=True)
        if s < best_S:
            best_S, best_values = s, vals

    ratio = best_S * np.sin(np.pi / nsc) / np.sin(np.pi / nsi)
    print("Final side length:", ratio)

    stem = f"{N}_{nsi}_in_{nsc}_jax_s{args.seed0}"
    file_i = 1
    while os.path.exists(f"{stem}.json"):
        stem = f"{N}_{nsi}_in_{nsc}_jax_s{args.seed0}_({file_i})"
        file_i += 1
    with open(f"{stem}.json", "w") as f:
        json.dump({
            "inner_polygons": N,
            "inner_sides": nsi,
            "container_sides": nsc,
            "container_circumradius": float(best_S),
            "side_length": float(ratio),
            "placements": [
                {"x": float(best_values[i * 3]), "y": float(best_values[i * 3 + 1]), "angle": float(best_values[i * 3 + 2])}
                for i in range(N)
            ],
        }, f, indent=2)
    print(f"Saved {stem}.json")
