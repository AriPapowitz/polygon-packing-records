"""GPU-batched polygon packer: thousands of simultaneous anneal restarts in JAX.

The whole search runs on-device with a leading batch axis:
  - penalty: vectorized SAT overlap + container violation (same math as
    polygon_packer.py / packer_jax.py, verified equivalent to 8e-16)
  - optimizer: natively-batched L-BFGS (two-loop recursion + Armijo
    backtracking, per-instance convergence masks)
  - anneal: per-instance shrink schedule with random-kick rescue (the batched
    equivalent of the original's basinhopping), instances retire independently

Workflow: run big batches here (GPU, float32 by default on consumer cards),
then certify survivors on CPU with validate_packing.py --polish --squeeze.

Usage:
    python packer_gpu.py 22 3 6 --batch 2048 --waves 4          # GPU exploration
    python packer_gpu.py 8 3 6 --batch 16 --waves 1 --x64      # CPU smoke test
Outputs {n}_{nsi}_in_{nsc}_gpu.json (+ _top{k} variants) for validate_packing.py.
"""

import argparse
import json
import os
import time

import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("inner_polygons", type=int)
arg_parser.add_argument("inner_sides", type=int)
arg_parser.add_argument("container_sides", type=int)
arg_parser.add_argument("--batch", type=int, default=2048, help="Restarts per wave (GPU-resident)")
arg_parser.add_argument("--waves", type=int, default=4, help="Number of waves (batches) to run")
arg_parser.add_argument("--tolerance", type=float, default=None,
                        help="Penalty acceptance (default 1e-8 for x64, 3e-8 for f32)")
arg_parser.add_argument("--finalstep", type=float, default=0.0001)
arg_parser.add_argument("--kicks", type=int, default=10, help="Random-kick rescues before an instance retires")
arg_parser.add_argument("--kick-size", type=float, default=0.1)
arg_parser.add_argument("--lbfgs-iters", type=int, default=250, help="Max L-BFGS iterations per shrink step")
arg_parser.add_argument("--max-outer", type=int, default=8000, help="Safety cap on shrink steps per wave")
arg_parser.add_argument("--seed0", type=int, default=0)
arg_parser.add_argument("--save-top", type=int, default=3, help="Save the k best packings per run")
arg_parser.add_argument("--x64", action="store_true", help="float64 (A100/H100/CPU); default float32")
args = arg_parser.parse_args()

import jax
# x64 is always enabled: the SEARCH runs in float32 by default (fast on consumer
# GPUs), but every wave's top survivors are re-polished and re-validated in
# float64 on-device before ranking (float32 acceptance alone lets ~1e-4-scale
# violations masquerade as record-beating sizes — learned the hard way).
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

DTYPE = jnp.float64 if args.x64 else jnp.float32
TOL = args.tolerance if args.tolerance is not None else (1e-8 if args.x64 else 3e-8)
TOL64 = 1e-24          # f64 refine: penalty below this counts as exactly valid
REFINE_TOP_K = 64      # survivors per wave that get the f64 treatment

N = args.inner_polygons
nsi = args.inner_sides
nsc = args.container_sides
D = N * 3
final_step = args.finalstep

# ---------------------------------------------------------------- geometry
inner_angles = np.linspace(0, 2 * np.pi, nsi, endpoint=False)
cont_angles = np.linspace(0, 2 * np.pi, nsc, endpoint=False)
CONT_APOTHEM = float(np.cos(np.pi / nsc))
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
LOWEST_S = float(np.sqrt(N * (nsi / 2 * np.sin(2 * np.pi / nsi)) / (nsc / 2 * np.sin(2 * np.pi / nsc))))
RATIO = float(np.sin(np.pi / nsc) / np.sin(np.pi / nsi))  # side ratio = S * RATIO


def make_batch_vg(dtype):
    """Build a batched value_and_grad of the penalty with all constants in `dtype`."""
    unit_vertices = jnp.asarray(np.column_stack((np.cos(inner_angles), np.sin(inner_angles))), dtype=dtype)
    unit_normals = jnp.asarray(np.column_stack((np.cos(inner_angles + np.pi / nsi), np.sin(inner_angles + np.pi / nsi))), dtype=dtype)
    cont_normals = jnp.asarray(np.column_stack((np.cos(cont_angles + np.pi / nsc), np.sin(cont_angles + np.pi / nsc))), dtype=dtype)

    def penalty(x, S):
        v = x.reshape(N, 3)
        xy, a = v[:, :2], v[:, 2]
        cos_a, sin_a = jnp.cos(a), jnp.sin(a)
        rot = jnp.stack([jnp.stack([cos_a, -sin_a], -1), jnp.stack([sin_a, cos_a], -1)], -2)
        verts = jnp.einsum("nij,vj->nvi", rot, unit_vertices) + xy[:, None, :]
        normals = jnp.einsum("nij,vj->nvi", rot, unit_normals)
        proj_wall = jnp.einsum("nvi,ci->nvc", verts, cont_normals)
        pen = jnp.sum(jnp.maximum(proj_wall - CONT_APOTHEM * S, 0.0) ** 2)
        axes = jnp.concatenate([normals[PAIR_I], normals[PAIR_J]], axis=1)
        proj_i = jnp.einsum("pvi,pai->pav", verts[PAIR_I], axes)
        proj_j = jnp.einsum("pvi,pai->pav", verts[PAIR_J], axes)
        overlap = (jnp.minimum(proj_i.max(-1), proj_j.max(-1))
                   - jnp.maximum(proj_i.min(-1), proj_j.min(-1)))
        pen += jnp.sum(jnp.maximum(overlap.min(-1), 0.0) ** 2)
        return pen

    return jax.vmap(jax.value_and_grad(penalty), in_axes=(0, 0))


batch_value_and_grad = make_batch_vg(DTYPE)      # search precision
batch_vg64 = make_batch_vg(jnp.float64)          # refine precision

# ---------------------------------------------------------------- batched L-BFGS
M_HIST = 8
C1 = 1e-4
GTOL = 1e-12


def batched_lbfgs(x, S, max_iters, vg=None):
    """Minimize penalty for every instance simultaneously. x: (B,D), S: (B,).
    Runs in the dtype of x; pass vg=batch_vg64 for float64 refinement."""
    if vg is None:
        vg = batch_value_and_grad
    dt = x.dtype
    B = x.shape[0]
    f, g = vg(x, S)
    state = {
        "x": x, "f": f, "g": g,
        "s_hist": jnp.zeros((M_HIST, B, D), dt),
        "y_hist": jnp.zeros((M_HIST, B, D), dt),
        "rho": jnp.zeros((M_HIST, B), dt),
        "head": jnp.array(0, jnp.int32),          # circular-buffer write index
        "active": jnp.ones((B,), bool),
        "it": jnp.array(0, jnp.int32),
    }

    def direction(state):
        """Two-loop recursion, batched. Zero-rho pairs are automatic no-ops."""
        q = state["g"]
        alphas = []
        for k in range(M_HIST):
            idx = (state["head"] - 1 - k) % M_HIST
            s_k, y_k, rho_k = state["s_hist"][idx], state["y_hist"][idx], state["rho"][idx]
            alpha = rho_k * jnp.einsum("bd,bd->b", s_k, q)
            q = q - alpha[:, None] * y_k
            alphas.append((idx, alpha))
        idx_last = (state["head"] - 1) % M_HIST
        sy = jnp.einsum("bd,bd->b", state["s_hist"][idx_last], state["y_hist"][idx_last])
        yy = jnp.einsum("bd,bd->b", state["y_hist"][idx_last], state["y_hist"][idx_last])
        gamma = jnp.where(yy > 0, sy / jnp.maximum(yy, 1e-30), 1.0)
        r = jnp.clip(gamma, 1e-4, 1e4)[:, None] * q
        for idx, alpha in reversed(alphas):
            s_k, y_k, rho_k = state["s_hist"][idx], state["y_hist"][idx], state["rho"][idx]
            beta = rho_k * jnp.einsum("bd,bd->b", y_k, r)
            r = r + (alpha - beta)[:, None] * s_k
        d = -r
        # descent safeguard: fall back to steepest descent where needed
        gd = jnp.einsum("bd,bd->b", state["g"], d)
        bad = gd >= 0
        d = jnp.where(bad[:, None], -state["g"], d)
        gd = jnp.where(bad, -jnp.einsum("bd,bd->b", state["g"], state["g"]), gd)
        return d, gd

    def body(state):
        d, gd = direction(state)
        # Armijo backtracking, batched: find per-instance step by halving
        t = jnp.ones((state["x"].shape[0],), dt)
        accepted = jnp.zeros_like(state["active"])
        best_t = jnp.zeros_like(t)

        def ls_body(j, carry):
            t, accepted, best_t = carry
            f_new, _ = vg(state["x"] + t[:, None] * d, S)
            ok = f_new <= state["f"] + C1 * t * gd
            newly = ok & ~accepted
            best_t = jnp.where(newly, t, best_t)
            accepted = accepted | ok
            t = jnp.where(accepted, t, t * 0.5)
            return t, accepted, best_t

        t, accepted, best_t = jax.lax.fori_loop(0, 25, ls_body, (t, accepted, best_t))
        step = jnp.where(accepted, best_t, 0.0)

        x_new = state["x"] + step[:, None] * d
        f_new, g_new = vg(x_new, S)
        improved = (f_new < state["f"]) & state["active"]
        x_next = jnp.where(improved[:, None], x_new, state["x"])
        f_next = jnp.where(improved, f_new, state["f"])
        g_next = jnp.where(improved[:, None], g_new, state["g"])

        # history update (only meaningful where improved; rho=0 disables elsewhere)
        s_vec = x_next - state["x"]
        y_vec = g_next - state["g"]
        sy = jnp.einsum("bd,bd->b", s_vec, y_vec)
        rho_new = jnp.where(improved & (sy > 1e-30), 1.0 / jnp.maximum(sy, 1e-30), 0.0)
        head = state["head"]
        s_hist = state["s_hist"].at[head].set(s_vec)
        y_hist = state["y_hist"].at[head].set(y_vec)
        rho = state["rho"].at[head].set(rho_new)

        gmax = jnp.max(jnp.abs(g_next), axis=1)
        still = improved & (gmax > GTOL) & (f_next > 0)
        return {
            "x": x_next, "f": f_next, "g": g_next,
            "s_hist": s_hist, "y_hist": y_hist, "rho": rho,
            "head": (head + 1) % M_HIST,
            "active": state["active"] & still,
            "it": state["it"] + 1,
        }

    def cond(state):
        return jnp.any(state["active"]) & (state["it"] < max_iters)

    state = jax.lax.while_loop(cond, body, state)
    return state["x"], state["f"]


# ---------------------------------------------------------------- batched anneal wave

def init_batch(key, B):
    """Half uniform-random, half jittered-grid starts (mirrors the CPU packer)."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    S0 = LOWEST_S * (2.0 + 2.0 * jax.random.uniform(k1, (B,), DTYPE))
    xy = jax.random.uniform(k2, (B, N, 2), DTYPE, -0.5, 0.5) * S0[:, None, None]
    ang = jax.random.uniform(k3, (B, N), DTYPE, 0.0, 2 * np.pi)

    side = int(np.ceil(np.sqrt(N)))
    lin = np.linspace(-0.45, 0.45, side)
    gx, gy = np.meshgrid(lin, lin)
    grid = jnp.asarray(np.column_stack((gx.ravel(), gy.ravel()))[:N], DTYPE)   # (N,2)
    grid_xy = grid[None, :, :] * S0[:, None, None]
    use_grid = jax.random.uniform(k4, (B,), DTYPE) < 0.5
    xy = jnp.where(use_grid[:, None, None], grid_xy, xy)
    xy = xy + jax.random.normal(k5, xy.shape, DTYPE) * 0.01

    x = jnp.concatenate([xy, ang[:, :, None]], axis=2).reshape(B, D)
    return x, S0


def wave(key, B):
    """One full wave: B instances annealed until all retire. Returns (S, x) per instance."""
    x, S = init_batch(key, B)
    state = {
        "x": x, "S": S,
        "initial_S": S,
        "best_x": x, "best_S": jnp.full((B,), jnp.inf, DTYPE),
        "kicks_left": jnp.full((B,), args.kicks, jnp.int32),
        "done": jnp.zeros((B,), bool),
        "key": key,
        "outer": jnp.array(0, jnp.int32),
    }

    def body(state):
        x_opt, pen = batched_lbfgs(state["x"], state["S"], args.lbfgs_iters)
        ok = (pen < TOL) & ~state["done"]

        span = jnp.maximum(state["initial_S"] - LOWEST_S, 1e-9)
        mult = 1 - final_step - (state["S"] - LOWEST_S) * (0.01 - final_step) / span
        mult = jnp.minimum(mult, 1 - final_step)  # guard: always shrink

        # success: record best, shrink, refill kicks
        best_S = jnp.where(ok, state["S"], state["best_S"])
        best_x = jnp.where(ok[:, None], x_opt, state["best_x"])
        S_next = jnp.where(ok, state["S"] * mult, state["S"])
        x_next = jnp.where(ok[:, None], x_opt * mult[:, None], x_opt)
        kicks = jnp.where(ok, args.kicks, state["kicks_left"])

        # failure with kicks left: random perturbation, retry at same S
        key, sub = jax.random.split(state["key"])
        noise = jax.random.normal(sub, x_opt.shape, DTYPE) * args.kick_size
        failed = ~ok & ~state["done"]
        kick = failed & (state["kicks_left"] > 0)
        x_next = jnp.where(kick[:, None], x_opt + noise, x_next)
        kicks = jnp.where(kick, kicks - 1, kicks)

        # failure with no kicks: retire
        done = state["done"] | (failed & (state["kicks_left"] <= 0))

        return {
            "x": x_next, "S": S_next, "initial_S": state["initial_S"],
            "best_x": best_x, "best_S": best_S,
            "kicks_left": kicks, "done": done, "key": key,
            "outer": state["outer"] + 1,
        }

    def cond(state):
        return jnp.any(~state["done"]) & (state["outer"] < args.max_outer)

    state = jax.lax.while_loop(cond, body, state)
    return state["best_S"], state["best_x"]


wave_jit = jax.jit(wave, static_argnums=(1,))

_lbfgs64 = jax.jit(lambda x, S, iters: batched_lbfgs(x, S, iters, vg=batch_vg64),
                   static_argnums=(2,))


def refine64(x, S):
    """Float64 on-device refinement of wave survivors.

    float32 acceptance lets ~1e-4-scale violations through, so raw wave sizes
    are optimistically wrong. Here each survivor is (1) re-polished in float64
    at its claimed S, (2) grown until a provably-feasible packing exists,
    (3) squeezed back down while feasibility holds. Returned sizes are honest:
    penalty(x64, S64) < TOL64.
    """
    x = jnp.asarray(x, jnp.float64)
    S = jnp.asarray(S, jnp.float64)

    x, pen = _lbfgs64(x, S, 400)
    for _ in range(60):                      # grow until feasible
        bad = pen >= TOL64
        if not bool(bad.any()):
            break
        S = jnp.where(bad, S * (1 + 5e-5), S)
        x2, pen2 = _lbfgs64(x, S, 400)
        x = jnp.where(bad[:, None], x2, x)
        pen = jnp.where(bad, pen2, pen)
    valid = pen < TOL64

    step = jnp.full(S.shape, 3e-5, jnp.float64)
    for _ in range(60):                      # squeeze down while feasible
        live = valid & (step >= 1e-9)
        if not bool(live.any()):
            break
        S_try = jnp.where(live, S * (1 - step), S)
        x_try, pen_try = _lbfgs64(x * (1 - step)[:, None], S_try, 400)
        ok = live & (pen_try < TOL64)
        S = jnp.where(ok, S_try, S)
        x = jnp.where(ok[:, None], x_try, x)
        step = jnp.where(ok, step, step / 4)
    return np.asarray(S), np.asarray(x), np.asarray(valid)


def save_solution(stem, S, x):
    out = f"{stem}.json"
    file_i = 1
    while os.path.exists(out):
        out = f"{stem}_({file_i}).json"
        file_i += 1
    with open(out, "w") as f:
        json.dump({
            "inner_polygons": N, "inner_sides": nsi, "container_sides": nsc,
            "container_circumradius": float(S),
            "side_length": float(S) * RATIO,
            "placements": [{"x": float(x[i * 3]), "y": float(x[i * 3 + 1]),
                            "angle": float(x[i * 3 + 2])} for i in range(N)],
        }, f, indent=2)
    return out


if __name__ == "__main__":
    print(f"device: {jax.devices()[0]}  dtype: {DTYPE.__name__}  tol: {TOL:g}")
    print(f"{N} {nsi}-gons in a {nsc}-gon; lower bound S = {LOWEST_S:.6f} (ratio {LOWEST_S * RATIO:.6f})")

    t0 = time.perf_counter()
    all_S, all_x = [], []
    for w in range(args.waves):
        key = jax.random.PRNGKey(args.seed0 + w)
        best_S, best_x = wave_jit(key, args.batch)
        best_S, best_x = np.asarray(best_S), np.asarray(best_x)
        finite = np.where(np.isfinite(best_S))[0]
        raw_best = best_S[finite].min() if len(finite) else float("inf")
        # float64 refinement of this wave's top survivors — the honest sizes
        if len(finite):
            top = finite[np.argsort(best_S[finite])[:REFINE_TOP_K]]
            S64, x64, ok64 = refine64(best_x[top], best_S[top])
            all_S.append(S64[ok64])
            all_x.append(x64[ok64])
            wave_best = S64[ok64].min() if ok64.any() else float("inf")
        else:
            wave_best = float("inf")
        print(f"wave {w}: {len(finite)}/{args.batch} valid, "
              f"f32 raw best = {raw_best * RATIO:.9f}, "
              f"f64 refined best = {wave_best * RATIO:.9f} "
              f"({time.perf_counter() - t0:.1f}s total)", flush=True)

    S_all = np.concatenate(all_S)
    x_all = np.concatenate(all_x)
    order = np.argsort(S_all)
    print(f"TOTAL: {len(S_all)} valid restarts in {time.perf_counter() - t0:.1f}s")
    print("Best ratio:", S_all[order[0]] * RATIO)
    for rank in range(min(args.save_top, len(order))):
        i = order[rank]
        out = save_solution(f"{N}_{nsi}_in_{nsc}_gpu_top{rank + 1}", S_all[i], x_all[i])
        print(f"  #{rank + 1}: ratio {S_all[i] * RATIO:.9f} -> {out}")
