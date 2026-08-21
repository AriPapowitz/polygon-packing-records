"""Structured basin-hopping packer: population search with an elite pool and
a portfolio of structured moves, on top of pack_core's batched GPU anneal.

Why: pure random multi-start converges whatever basin it lands in but never
finds globally-rearranged optima (measured: 4,700+ restarts never found the
tri-in-hex n=21 record). The literature's winning recipe (ImprovEvolve, GVS,
Specht MBS) is: seed from structure, refine, then perturb with MOVES that
change the packing's combinatorics — vacancy teleports, coherent cluster
tilts, row shears — and re-refine, keeping an elite pool.

Round loop:
  starts (elites x structured moves, + lattice/random immigrants)
    -> batched shrink-anneal (f32 on GPU)  -> f64 refine of the round's best
    -> elite pool update -> next round's starts

Usage (run from a results dir; multi-GPU = one process per CUDA_VISIBLE_DEVICES):
    python ../packer_bh.py 44 3 5 --batch 1024 --rounds 40 --target 3.52139
    python ../packer_bh.py 44 3 5 --seed-json "44_drop_top*.json" --batch 2048
"""

import argparse
import glob
import json
import time

import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("inner_polygons", type=int)
arg_parser.add_argument("inner_sides", type=int)
arg_parser.add_argument("container_sides", type=int)
arg_parser.add_argument("--batch", type=int, default=1024)
arg_parser.add_argument("--rounds", type=int, default=30)
arg_parser.add_argument("--elites", type=int, default=24)
arg_parser.add_argument("--refine-top", type=int, default=32, help="f64 refines per round")
arg_parser.add_argument("--immigrants", type=float, default=0.25,
                        help="fraction of each round restarted from lattice/random")
arg_parser.add_argument("--seed-json", default=None, help="glob of solution JSONs to seed elites")
arg_parser.add_argument("--target", type=float, default=None,
                        help="stop when certified side ratio <= this")
arg_parser.add_argument("--tolerance", type=float, default=None)
arg_parser.add_argument("--kicks", type=int, default=6)
arg_parser.add_argument("--kick-size", type=float, default=0.1)
arg_parser.add_argument("--lbfgs-iters", type=int, default=250)
arg_parser.add_argument("--max-outer", type=int, default=4000)
arg_parser.add_argument("--uplift", type=float, default=0.012,
                        help="mean relative S headroom given to perturbed elites")
arg_parser.add_argument("--seed0", type=int, default=0)
arg_parser.add_argument("--save-top", type=int, default=3)
arg_parser.add_argument("--x64", action="store_true")


def main(argv=None):
    args = arg_parser.parse_args(argv)

    import jax
    import jax.numpy as jnp

    from . import pack_core

    DTYPE = jnp.float64 if args.x64 else jnp.float32
    TOL = args.tolerance if args.tolerance is not None else (1e-8 if args.x64 else 3e-8)
    N, nsi, nsc = args.inner_polygons, args.inner_sides, args.container_sides
    D = N * 3
    rng = np.random.default_rng(args.seed0)

    eng = pack_core.Engine(N, nsi, nsc, dtype=DTYPE)
    anneal = eng.make_anneal(TOL, kicks=args.kicks, kick_size=args.kick_size,
                             lbfgs_iters=args.lbfgs_iters, max_outer=args.max_outer)

    SIDE = 2 * np.sin(np.pi / nsi)            # inner polygon side, circumradius units
    SPACING = SIDE if nsi != 3 else 1.0       # typical center-center distance


    # ---------------------------------------------------------------- seeding
    def lattice_starts(k):
        """k lattice fills: triangle lattice for 3-gons, grid for 4-, honeycomb for
        6-gons, jittered grid otherwise; random global rotation/offset each."""
        out = []
        for _ in range(k):
            pts, angs = [], []
            if nsi == 3:
                pitch, row_h = SIDE / 2, 1.5
                for r in range(-12, 13):
                    for c in range(-12, 13):
                        up = (c % 2 == 0)
                        pts.append([c * pitch + (r % 2) * pitch,
                                    r * row_h + (0.5 if up else 1.0)])
                        angs.append(np.pi / 2 if up else -np.pi / 2)
            elif nsi == 6:
                for r in range(-8, 9):
                    for c in range(-8, 9):
                        pts.append([c * 1.5, (r + (c % 2) / 2) * np.sqrt(3)])
                        angs.append(0.0)
            else:
                step = SIDE * (np.sqrt(2) if nsi == 4 else 1.05)
                for r in range(-8, 9):
                    for c in range(-8, 9):
                        pts.append([c * step, r * step])
                        angs.append(np.pi / nsi if nsi == 4 else rng.uniform(0, 2 * np.pi))
            pts = np.array(pts)
            angs = np.array(angs)
            th = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            pts = pts @ R.T + rng.uniform(-0.5, 0.5, 2) * SPACING
            angs = angs + th
            keep = np.argsort(np.abs(pts).max(1))[:N]         # most central n cells
            x = np.column_stack([pts[keep], angs[keep]]).ravel()
            x[::3] += rng.normal(0, 0.01, N)
            x[1::3] += rng.normal(0, 0.01, N)
            verts_r = np.abs(pts[keep]).max() + 1.0
            S = max(verts_r / eng.apothem, eng.lowest_S * 1.01) * rng.uniform(1.0, 1.15)
            out.append((x, S))
        return out


    def random_starts(k):
        out = []
        for _ in range(k):
            S = eng.lowest_S * rng.uniform(2.0, 4.0)
            xy = rng.uniform(-0.5, 0.5, (N, 2)) * S
            ang = rng.uniform(0, 2 * np.pi, N)
            out.append((np.column_stack([xy, ang]).ravel(), S))
        return out


    # ---------------------------------------------------------------- moves
    def mv_teleport(x, S):
        """Vacancy move: shape whose neighborhood is most crowded teleports to the
        emptiest sampled spot (Huang & Ye's greedy vacancy idea, stochastic form)."""
        v = x.reshape(N, 3).copy()
        d = np.linalg.norm(v[None, :, :2] - v[:, None, :2], axis=-1) + np.eye(N) * 1e9
        src = np.argmin(d.min(1) + rng.normal(0, 0.05 * SPACING, N))
        cand = rng.uniform(-1, 1, (128, 2)) * S * eng.apothem * 0.95
        cd = np.linalg.norm(cand[:, None] - np.delete(v[:, :2], src, 0)[None], axis=-1).min(1)
        v[src, :2] = cand[np.argmax(cd)]
        v[src, 2] = rng.uniform(0, 2 * np.pi)
        return v.ravel()

    def mv_cluster_tilt(x, S):
        """Rotate a contiguous cluster coherently around its own centroid
        (AlphaEvolve's winning hex-in-hex motif was per-cluster tilts)."""
        v = x.reshape(N, 3).copy()
        c = v[rng.integers(N), :2]
        r = rng.uniform(1.0, 2.5) * SPACING
        sel = np.linalg.norm(v[:, :2] - c, axis=1) < r
        th = rng.uniform(0.1, 0.45) * rng.choice([-1, 1])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pivot = v[sel, :2].mean(0)
        v[sel, :2] = (v[sel, :2] - pivot) @ R.T + pivot
        v[sel, 2] += th
        return v.ravel()

    def mv_shear(x, S):
        """Slide everything on one side of a random line by a fraction of a spacing
        (row-shift / lattice-defect move)."""
        v = x.reshape(N, 3).copy()
        phi = rng.uniform(0, np.pi)
        nvec = np.array([np.cos(phi), np.sin(phi)])
        side = (v[:, :2] - v[:, :2].mean(0)) @ nvec > rng.normal(0, 0.5) * SPACING
        tvec = np.array([-nvec[1], nvec[0]])
        v[side, :2] += tvec * rng.uniform(0.15, 0.6) * SPACING
        return v.ravel()

    def mv_local_shake(x, S):
        v = x.reshape(N, 3).copy()
        c = v[rng.integers(N), :2]
        sel = np.linalg.norm(v[:, :2] - c, axis=1) < rng.uniform(1.0, 2.0) * SPACING
        v[sel, :2] += rng.normal(0, 0.12 * SPACING, (sel.sum(), 2))
        v[sel, 2] += rng.normal(0, 0.3, sel.sum())
        return v.ravel()

    def mv_angle_scramble(x, S):
        v = x.reshape(N, 3).copy()
        sel = rng.random(N) < rng.uniform(0.1, 0.35)
        v[sel, 2] = rng.uniform(0, 2 * np.pi, sel.sum())
        return v.ravel()

    def mv_global_jiggle(x, S):
        return x + rng.normal(0, 0.05 * SPACING, D)

    MOVES = [mv_teleport, mv_cluster_tilt, mv_shear, mv_local_shake,
             mv_angle_scramble, mv_global_jiggle]


    # ---------------------------------------------------------------- elite pool
    class Pool:
        def __init__(self, cap):
            self.cap = cap
            self.items = []                                  # (S_certified, x) sorted

        def add(self, S, x):
            for S0, _ in self.items:
                if abs(S0 - S) < 1e-7:                       # same basin, keep first
                    return False
            self.items.append((S, x))
            self.items.sort(key=lambda t: t[0])
            del self.items[self.cap:]
            return True

        def sample(self, k):
            m = len(self.items)
            w = 1.0 / (1.0 + np.arange(m))                   # rank-weighted
            idx = rng.choice(m, size=k, p=w / w.sum())
            return [self.items[i] for i in idx]


    pool = Pool(args.elites)
    if args.seed_json:
        for p in sorted(glob.glob(args.seed_json)):
            sol, x, S = pack_core.load_solution(p)
            if sol["inner_polygons"] != N:
                print(f"  (skip {p}: n={sol['inner_polygons']})")
                continue
            S64, x64, ok = eng.refine64(x[None], np.array([S]), iters=800,
                                        grow_rounds=40, squeeze_rounds=120)
            if ok[0]:
                pool.add(float(S64[0]), x64[0])
                print(f"  seeded elite from {p}: ratio {S64[0] * eng.ratio:.9f}")

    anneal_jit = jax.jit(lambda key, x, S: anneal(key, x, S))


    def build_round_starts(B):
        xs, Ss = [], []
        n_imm = int(B * args.immigrants) if pool.items else B
        n_elite = B - n_imm
        if n_elite:
            for S_e, x_e in pool.sample(n_elite):
                mv = MOVES[rng.integers(len(MOVES))]
                uplift = 1.0 + rng.exponential(args.uplift)
                xs.append(mv(x_e, S_e))
                Ss.append(S_e * uplift)
        imm = lattice_starts((n_imm + 1) // 2) + random_starts(n_imm // 2)
        for x_i, S_i in imm:
            xs.append(x_i)
            Ss.append(S_i)
        return (jnp.asarray(np.stack(xs), DTYPE), jnp.asarray(np.array(Ss), DTYPE))


    if True:  # CLI body
        print(f"device: {jax.devices()[0]}  dtype: {DTYPE.__name__}  "
              f"n={N} {nsi}-gons in {nsc}-gon  lower bound ratio "
              f"{eng.lowest_S * eng.ratio:.6f}")
        t0 = time.perf_counter()
        best_ever = np.inf
        for rd in range(args.rounds):
            x0, S0 = build_round_starts(args.batch)
            key = jax.random.PRNGKey(args.seed0 * 100003 + rd)
            bS, bx = anneal_jit(key, x0, S0)
            bS, bx = np.asarray(bS, float), np.asarray(bx, float)
            fin = np.where(np.isfinite(bS))[0]
            if len(fin) == 0:
                print(f"round {rd}: no valid anneals")
                continue
            top = fin[np.argsort(bS[fin])[:args.refine_top]]
            S64, x64, ok = eng.refine64(bx[top], bS[top])
            added = 0
            for s, xx, o in zip(S64, x64, ok):
                if o:
                    added += pool.add(float(s), xx)
            best = pool.items[0][0] if pool.items else np.inf
            print(f"round {rd}: {len(fin)}/{args.batch} valid, raw {bS[fin].min() * eng.ratio:.9f}, "
                  f"certified best {best * eng.ratio:.9f} "
                  f"({added} new elites, pool {len(pool.items)}, "
                  f"{time.perf_counter() - t0:.0f}s)", flush=True)
            if best < best_ever:
                best_ever = best
                for rank, (s, xx) in enumerate(pool.items[:args.save_top]):
                    pack_core.save_solution(
                        f"{N}_{nsi}_in_{nsc}_bh_top{rank + 1}.json", N, nsi, nsc, s, xx,
                        extra={"method": "structured basin-hopping (packer_bh)"})
            if args.target and best * eng.ratio <= args.target:
                print(f"TARGET reached: {best * eng.ratio:.9f} <= {args.target}")
                break
        if pool.items:
            print(f"BEST: ratio {pool.items[0][0] * eng.ratio:.9f} "
                  f"({time.perf_counter() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
