"""Turn a valid n-packing into the best (n-1)-packing by removal + squeeze.

For each of the n removal choices, drop that shape, then f64 polish/grow/
squeeze the remaining n-1 (all removals refined as one batch). The winner is
the monotonicity construction: any (n-1)-packing inherits the n-packing's
container, and the squeeze then shrinks it further into the freed space.

Usage:
    python drop_one.py results/45_recon_refined.json --out-prefix results/44_drop
"""

import argparse

import numpy as np

import pack_core

parser = argparse.ArgumentParser()
parser.add_argument("solution", help="valid n-packing JSON")
parser.add_argument("--out-prefix", default=None)
parser.add_argument("--save-top", type=int, default=3)
parser.add_argument("--iters", type=int, default=800)
parser.add_argument("--squeeze-rounds", type=int, default=300)
parser.add_argument("--squeeze-step", type=float, default=2e-4)
args = parser.parse_args()

sol, x, S = pack_core.load_solution(args.solution)
n, nsi, nsc = sol["inner_polygons"], sol["inner_sides"], sol["container_sides"]
m = n - 1
eng = pack_core.Engine(m, nsi, nsc)

drops = np.stack([np.delete(x.reshape(n, 3), i, axis=0).ravel() for i in range(n)])
S0 = np.full(n, S, float)
print(f"{n} removal candidates -> squeezing {m} {nsi}-gons in a {nsc}-gon "
      f"from S={S:.6f} (ratio {S * eng.ratio:.6f})")

S64, x64, ok = eng.refine64(drops, S0, iters=args.iters,
                            grow_rounds=20, squeeze_rounds=args.squeeze_rounds,
                            squeeze_step=args.squeeze_step)
if not ok.any():
    raise SystemExit("no removal refined to a valid packing — check the input")

order = np.argsort(np.where(ok, S64, np.inf))
prefix = args.out_prefix or f"{m}_{nsi}_in_{nsc}_drop"
print(f"\nbest removals (of {int(ok.sum())} valid):")
for rank in range(min(args.save_top, int(ok.sum()))):
    i = order[rank]
    out = pack_core.save_solution(
        f"{prefix}_top{rank + 1}.json", m, nsi, nsc, S64[i], x64[i],
        extra={"method": f"drop shape {int(i)} from {args.solution} + f64 squeeze"})
    print(f"  #{rank + 1}: removed {i:2d} -> ratio {S64[i] * eng.ratio:.9f}  ({out})")
