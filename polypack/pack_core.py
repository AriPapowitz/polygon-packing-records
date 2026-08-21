"""Shared batched packing engine: penalty, L-BFGS, anneal, f64 refine.

This is packer_gpu.py's proven machinery factored into an importable,
(n, nsi, nsc)-parameterized form so multiple drivers (packer_bh.py,
drop_one.py) can share it. packer_gpu.py itself is left untouched.

Conventions match the whole pipeline: container regular nsc-gon with a vertex
at angle 0 and circumradius S; inner unit-circumradius nsi-gons placed by
(x, y, angle); side ratio s = S*sin(pi/nsc)/sin(pi/nsi).
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

M_HIST = 8
C1 = 1e-4
GTOL = 1e-12
TOL64 = 1e-24


class Engine:
    def __init__(self, n, nsi, nsc, dtype=jnp.float32):
        self.n, self.nsi, self.nsc = n, nsi, nsc
        self.D = n * 3
        self.dtype = dtype
        self.apothem = float(np.cos(np.pi / nsc))
        self.ratio = float(np.sin(np.pi / nsc) / np.sin(np.pi / nsi))
        self.lowest_S = float(np.sqrt(
            n * (nsi / 2 * np.sin(2 * np.pi / nsi))
            / (nsc / 2 * np.sin(2 * np.pi / nsc))))
        self.pair_i, self.pair_j = np.triu_indices(n, k=1)
        self.vg = self._make_vg(dtype)
        self.vg64 = self.vg if dtype == jnp.float64 else self._make_vg(jnp.float64)
        self._lbfgs64 = jax.jit(
            lambda x, S, iters: self.lbfgs(x, S, iters, vg=self.vg64),
            static_argnums=(2,))

    def _make_vg(self, dtype):
        n, nsi, nsc = self.n, self.nsi, self.nsc
        ia = np.linspace(0, 2 * np.pi, nsi, endpoint=False)
        ca = np.linspace(0, 2 * np.pi, nsc, endpoint=False)
        uv = jnp.asarray(np.column_stack((np.cos(ia), np.sin(ia))), dtype=dtype)
        un = jnp.asarray(np.column_stack((np.cos(ia + np.pi / nsi),
                                          np.sin(ia + np.pi / nsi))), dtype=dtype)
        cn = jnp.asarray(np.column_stack((np.cos(ca + np.pi / nsc),
                                          np.sin(ca + np.pi / nsc))), dtype=dtype)
        pi_, pj_ = self.pair_i, self.pair_j
        apothem = self.apothem

        def penalty(x, S):
            v = x.reshape(n, 3)
            xy, a = v[:, :2], v[:, 2]
            cos_a, sin_a = jnp.cos(a), jnp.sin(a)
            rot = jnp.stack([jnp.stack([cos_a, -sin_a], -1),
                             jnp.stack([sin_a, cos_a], -1)], -2)
            verts = jnp.einsum("nij,vj->nvi", rot, uv) + xy[:, None, :]
            normals = jnp.einsum("nij,vj->nvi", rot, un)
            proj_wall = jnp.einsum("nvi,ci->nvc", verts, cn)
            pen = jnp.sum(jnp.maximum(proj_wall - apothem * S, 0.0) ** 2)
            axes = jnp.concatenate([normals[pi_], normals[pj_]], axis=1)
            proj_i = jnp.einsum("pvi,pai->pav", verts[pi_], axes)
            proj_j = jnp.einsum("pvi,pai->pav", verts[pj_], axes)
            overlap = (jnp.minimum(proj_i.max(-1), proj_j.max(-1))
                       - jnp.maximum(proj_i.min(-1), proj_j.min(-1)))
            pen += jnp.sum(jnp.maximum(overlap.min(-1), 0.0) ** 2)
            return pen

        return jax.vmap(jax.value_and_grad(penalty), in_axes=(0, 0))

    # ------------------------------------------------------------ L-BFGS
    def lbfgs(self, x, S, max_iters, vg=None):
        """Batched minimize; x: (B,D), S: (B,). Runs in x.dtype."""
        if vg is None:
            vg = self.vg
        dt = x.dtype
        B, D = x.shape
        f, g = vg(x, S)
        state = {
            "x": x, "f": f, "g": g,
            "s_hist": jnp.zeros((M_HIST, B, D), dt),
            "y_hist": jnp.zeros((M_HIST, B, D), dt),
            "rho": jnp.zeros((M_HIST, B), dt),
            "head": jnp.array(0, jnp.int32),
            "active": jnp.ones((B,), bool),
            "it": jnp.array(0, jnp.int32),
        }

        def direction(st):
            q = st["g"]
            alphas = []
            for k in range(M_HIST):
                idx = (st["head"] - 1 - k) % M_HIST
                s_k, y_k, rho_k = st["s_hist"][idx], st["y_hist"][idx], st["rho"][idx]
                alpha = rho_k * jnp.einsum("bd,bd->b", s_k, q)
                q = q - alpha[:, None] * y_k
                alphas.append((idx, alpha))
            idx_last = (st["head"] - 1) % M_HIST
            sy = jnp.einsum("bd,bd->b", st["s_hist"][idx_last], st["y_hist"][idx_last])
            yy = jnp.einsum("bd,bd->b", st["y_hist"][idx_last], st["y_hist"][idx_last])
            gamma = jnp.where(yy > 0, sy / jnp.maximum(yy, 1e-30), 1.0)
            r = jnp.clip(gamma, 1e-4, 1e4)[:, None] * q
            for idx, alpha in reversed(alphas):
                s_k, y_k, rho_k = st["s_hist"][idx], st["y_hist"][idx], st["rho"][idx]
                beta = rho_k * jnp.einsum("bd,bd->b", y_k, r)
                r = r + (alpha - beta)[:, None] * s_k
            d = -r
            gd = jnp.einsum("bd,bd->b", st["g"], d)
            bad = gd >= 0
            d = jnp.where(bad[:, None], -st["g"], d)
            gd = jnp.where(bad, -jnp.einsum("bd,bd->b", st["g"], st["g"]), gd)
            return d, gd

        def body(st):
            d, gd = direction(st)
            t = jnp.ones((st["x"].shape[0],), dt)
            accepted = jnp.zeros_like(st["active"])
            best_t = jnp.zeros_like(t)

            def ls_body(j, carry):
                t, accepted, best_t = carry
                f_new, _ = vg(st["x"] + t[:, None] * d, S)
                ok = f_new <= st["f"] + C1 * t * gd
                newly = ok & ~accepted
                best_t = jnp.where(newly, t, best_t)
                accepted = accepted | ok
                t = jnp.where(accepted, t, t * 0.5)
                return t, accepted, best_t

            t, accepted, best_t = jax.lax.fori_loop(0, 25, ls_body, (t, accepted, best_t))
            step = jnp.where(accepted, best_t, 0.0)

            x_new = st["x"] + step[:, None] * d
            f_new, g_new = vg(x_new, S)
            improved = (f_new < st["f"]) & st["active"]
            x_next = jnp.where(improved[:, None], x_new, st["x"])
            f_next = jnp.where(improved, f_new, st["f"])
            g_next = jnp.where(improved[:, None], g_new, st["g"])

            s_vec = x_next - st["x"]
            y_vec = g_next - st["g"]
            sy = jnp.einsum("bd,bd->b", s_vec, y_vec)
            rho_new = jnp.where(improved & (sy > 1e-30),
                                1.0 / jnp.maximum(sy, 1e-30), 0.0)
            head = st["head"]
            gmax = jnp.max(jnp.abs(g_next), axis=1)
            still = improved & (gmax > GTOL) & (f_next > 0)
            return {
                "x": x_next, "f": f_next, "g": g_next,
                "s_hist": st["s_hist"].at[head].set(s_vec),
                "y_hist": st["y_hist"].at[head].set(y_vec),
                "rho": st["rho"].at[head].set(rho_new),
                "head": (head + 1) % M_HIST,
                "active": st["active"] & still,
                "it": st["it"] + 1,
            }

        def cond(st):
            return jnp.any(st["active"]) & (st["it"] < max_iters)

        state = jax.lax.while_loop(cond, body, state)
        return state["x"], state["f"]

    # ------------------------------------------------------------ anneal
    def make_anneal(self, tol, final_step=1e-4, kicks=10, kick_size=0.1,
                    lbfgs_iters=250, max_outer=8000):
        """Jitted shrink-anneal from given (x, S) until all instances retire.
        Returns fn(key, x, S) -> (best_S, best_x); failures keep S=inf."""

        def anneal(key, x, S):
            state = {
                "x": x, "S": S, "initial_S": S,
                "best_x": x, "best_S": jnp.full(S.shape, jnp.inf, x.dtype),
                "kicks_left": jnp.full(S.shape, kicks, jnp.int32),
                "done": jnp.zeros(S.shape, bool),
                "key": key, "outer": jnp.array(0, jnp.int32),
            }

            def body(st):
                x_opt, pen = self.lbfgs(st["x"], st["S"], lbfgs_iters)
                ok = (pen < tol) & ~st["done"]
                span = jnp.maximum(st["initial_S"] - self.lowest_S, 1e-9)
                mult = 1 - final_step - (st["S"] - self.lowest_S) * (0.01 - final_step) / span
                mult = jnp.minimum(mult, 1 - final_step)
                best_S = jnp.where(ok, st["S"], st["best_S"])
                best_x = jnp.where(ok[:, None], x_opt, st["best_x"])
                S_next = jnp.where(ok, st["S"] * mult, st["S"])
                x_next = jnp.where(ok[:, None], x_opt * mult[:, None], x_opt)
                kicks_n = jnp.where(ok, kicks, st["kicks_left"])
                key, sub = jax.random.split(st["key"])
                noise = jax.random.normal(sub, x_opt.shape, x_opt.dtype) * kick_size
                failed = ~ok & ~st["done"]
                kick = failed & (st["kicks_left"] > 0)
                x_next = jnp.where(kick[:, None], x_opt + noise, x_next)
                kicks_n = jnp.where(kick, kicks_n - 1, kicks_n)
                done = st["done"] | (failed & (st["kicks_left"] <= 0))
                return {"x": x_next, "S": S_next, "initial_S": st["initial_S"],
                        "best_x": best_x, "best_S": best_S,
                        "kicks_left": kicks_n, "done": done, "key": key,
                        "outer": st["outer"] + 1}

            def cond(st):
                return jnp.any(~st["done"]) & (st["outer"] < max_outer)

            state = jax.lax.while_loop(cond, body, state)
            return state["best_S"], state["best_x"]

        return jax.jit(anneal)

    # ------------------------------------------------------------ f64 refine
    def refine64(self, x, S, iters=400, grow_rounds=60, squeeze_rounds=60,
                 squeeze_step=3e-5, grow_rate=5e-5):
        """Polish -> grow-to-feasible -> squeeze, all in float64 on device.
        Returns (S, x, valid) as numpy; sizes satisfy penalty < TOL64."""
        x = jnp.asarray(x, jnp.float64)
        S = jnp.asarray(S, jnp.float64)
        x, pen = self._lbfgs64(x, S, iters)
        for _ in range(grow_rounds):
            bad = pen >= TOL64
            if not bool(bad.any()):
                break
            S = jnp.where(bad, S * (1 + grow_rate), S)
            x2, pen2 = self._lbfgs64(x, S, iters)
            x = jnp.where(bad[:, None], x2, x)
            pen = jnp.where(bad, pen2, pen)
        valid = pen < TOL64
        step = jnp.full(S.shape, squeeze_step, jnp.float64)
        for _ in range(squeeze_rounds):
            live = valid & (step >= 1e-9)
            if not bool(live.any()):
                break
            S_try = jnp.where(live, S * (1 - step), S)
            x_try, pen_try = self._lbfgs64(x * (1 - step)[:, None], S_try, iters)
            ok = live & (pen_try < TOL64)
            S = jnp.where(ok, S_try, S)
            x = jnp.where(ok[:, None], x_try, x)
            step = jnp.where(ok, step, step / 4)
        return np.asarray(S), np.asarray(x), np.asarray(valid)


# ---------------------------------------------------------------- helpers
def load_solution(path):
    import json
    with open(path) as f:
        sol = json.load(f)
    x = np.array([[p["x"], p["y"], p["angle"]] for p in sol["placements"]],
                 float).ravel()
    return sol, x, float(sol["container_circumradius"])


def save_solution(path, n, nsi, nsc, S, x, extra=None):
    import json
    ratio = float(np.sin(np.pi / nsc) / np.sin(np.pi / nsi))
    doc = {
        "inner_polygons": n, "inner_sides": nsi, "container_sides": nsc,
        "container_circumradius": float(S),
        "side_length": float(S) * ratio,
        "placements": [{"x": float(x[i * 3]), "y": float(x[i * 3 + 1]),
                        "angle": float(x[i * 3 + 2])} for i in range(n)],
    }
    if extra:
        doc.update(extra)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path
