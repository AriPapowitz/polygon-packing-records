"""Construct the exact 24-triangle lattice packing of a side-2 hexagon (s = 2.000),
the configuration behind the current trivial records for n = 22, 23 triangles in a
hexagon. Writes 24_3_in_6_lattice.json in polygon_packer conventions:
  - inner triangle: circumradius 1 (side sqrt(3)), vertices at angles 0/120/240 deg
  - container hexagon: circumradius S, vertices at angles 0/60/...; S = 2*sqrt(3) here
  - "up" triangle = rotation +pi/2, "down" = -pi/2

Also supports emitting n-removed variants for warm-starting record hunts:
    python build_lattice.py                 # writes the full 24-lattice
    python build_lattice.py --remove 3 17   # writes a 22-triangle variant
"""

import argparse
import json

import numpy as np

A = np.sqrt(3.0)          # triangle side length (circumradius 1)
S = 2 * np.sqrt(3.0)      # hexagon circumradius for side ratio exactly 2
UP, DOWN = np.pi / 2, -np.pi / 2


def lattice_placements():
    """24 (x, y, angle) triangles tiling the hexagon exactly. Rows top to bottom."""
    placements = []
    # top half: row bands y in [1.5, 3] and [0, 1.5]
    top_half = []
    # band [1.5, 3]: 3 up (base y=1.5), 2 down (base y=3)
    top_half += [(-1.5 * A + A / 2 + k * A, 2.0, UP) for k in range(3)]
    top_half += [(-A + A / 2 + k * A, 2.5, DOWN) for k in range(2)]
    # band [0, 1.5]: 4 up (base y=0), 3 down (base y=1.5)
    top_half += [(-2 * A + A / 2 + k * A, 0.5, UP) for k in range(4)]
    top_half += [(-1.5 * A + A / 2 + k * A, 1.0, DOWN) for k in range(3)]
    placements += top_half
    # bottom half: mirror y -> -y (flips orientation)
    placements += [(x, -y, -angle) for (x, y, angle) in top_half]
    return placements


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", type=int, nargs="*", default=[],
                        help="Indices (0-23) of lattice triangles to remove")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    placements = lattice_placements()
    assert len(placements) == 24
    keep = [p for i, p in enumerate(placements) if i not in set(args.remove)]
    n = len(keep)

    out = args.out or (f"{n}_3_in_6_lattice.json" if not args.remove
                       else f"{n}_3_in_6_lattice_rm{'-'.join(map(str, sorted(args.remove)))}.json")
    with open(out, "w") as f:
        json.dump({
            "inner_polygons": n,
            "inner_sides": 3,
            "container_sides": 6,
            "container_circumradius": S,
            "side_length": S * np.sin(np.pi / 6) / np.sin(np.pi / 3),
            "placements": [{"x": x, "y": y, "angle": a} for (x, y, a) in keep],
        }, f, indent=2)
    print(f"Wrote {out}  (n={n}, side ratio = {S / A})")


if __name__ == "__main__":
    main()
