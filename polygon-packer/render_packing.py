"""Render a packing solution .json to PNG/SVG.

Usage:
    python render_packing.py solution.json [-o out.png]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as ppt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--no-title", action="store_true",
                    help="omit the baked-in title (for paper figures)")
    args = ap.parse_args()

    with open(args.solution) as f:
        sol = json.load(f)
    n, nsi, nsc = sol["inner_polygons"], sol["inner_sides"], sol["container_sides"]
    S = sol["container_circumradius"]

    inner_angles = np.linspace(0, 2 * np.pi, nsi, endpoint=False)
    unit = np.column_stack((np.cos(inner_angles), np.sin(inner_angles)))
    cont_angles = np.linspace(0, 2 * np.pi, nsc, endpoint=False)
    cont = np.column_stack((np.cos(cont_angles), np.sin(cont_angles))) * S

    fig, ax = ppt.subplots(figsize=(6.4, 6.4))
    ax.plot(*np.vstack((cont, cont[:1])).T, color="black", linewidth=0.8)
    for p in sol["placements"]:
        c, s = np.cos(p["angle"]), np.sin(p["angle"])
        verts = unit @ np.array([[c, s], [-s, c]]) + [p["x"], p["y"]]
        ax.fill(*np.vstack((verts, verts[:1])).T, "#CCCCCC", edgecolor="black", linewidth=0.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ratio = sol.get("certified_side_length", sol["side_length"])
    if not args.no_title:
        ppt.title(f"n={n}: side = {ratio:.12f}")

    out = args.out or os.path.splitext(args.solution)[0] + ".png"
    ppt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
