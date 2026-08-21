"""How far did the harvest move the incumbents' configurations?

For every credited harvest claim whose archived reconstruction exists in
polygon-packer/results/, compare the reconstruction (the incumbent's
configuration, recovered from their published image) with the submitted
solution (after float64 refinement and squeeze). Shapes are matched by
optimal assignment on center distances; the reported displacement is the
per-shape center distance in engine units (inner circumradius = 1), with the
inner side length given for scale.

If refinement merely settles the incumbent's arrangement (removes slack
without changing the architecture), displacements stay far below the
center-to-center spacing; a genuine rearrangement moves at least one shape
by a large fraction of the spacing.

Run:  polygon-packer/.venv/Scripts/python paper/harvest_displacement.py
Output: paper/harvest_displacement.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "polygon-packer" / "results"
SOL = ROOT / "paper" / "solutions"
OUT = ROOT / "paper" / "harvest_displacement.csv"


def centers(path):
    doc = json.loads(Path(path).read_text())
    xy = np.array([[p["x"], p["y"]] for p in doc["placements"]], float)
    return doc, xy


def main():
    ledger = list(csv.DictReader(open(ROOT / "paper" / "ledger.csv",
                                      encoding="utf-8")))
    rows = []
    for r in ledger:
        if r["mechanism"] != "harvest" or r["credit_verified_in_scrape"] in ("", "—"):
            continue
        cat, n = r["category"], int(r["n"])
        recon = RES / f"sweep_{cat}_{n}.json"
        if not recon.exists():
            cands = sorted(RES.glob(f"*{cat}*{n}*recon*.json"))
            recon = cands[0] if cands else None
        if recon is None:
            rows.append([cat, n, "", "", "", "", "no archived reconstruction"])
            continue
        doc_r, xy_r = centers(recon)
        doc_s, xy_s = centers(SOL / f"{cat}_{n:02d}.json")
        if len(xy_r) != len(xy_s):
            rows.append([cat, n, "", "", "", "", "shape-count mismatch"])
            continue
        d = np.linalg.norm(xy_r[:, None] - xy_s[None, :], axis=-1)
        ri, ci = linear_sum_assignment(d)
        disp = d[ri, ci]
        side = 2 * np.sin(np.pi / doc_s["inner_sides"])
        rows.append([cat, n, f"{disp.max():.4f}", f"{np.median(disp):.4f}",
                     f"{side:.4f}", f"{disp.max() / side:.3f}", recon.name])

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n", "max_disp", "median_disp",
                    "inner_side", "max_disp_over_side", "reconstruction"])
        w.writerows(rows)

    done = [r for r in rows if r[2] != ""]
    frac = np.array([float(r[5]) for r in done])
    print(f"{len(done)} of {len(rows)} credited harvest claims compared")
    print(f"max displacement / inner side: median {np.median(frac):.3f}, "
          f"max {frac.max():.3f}")
    print(f"claims with max displacement > 0.5 side: {(frac > 0.5).sum()}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
