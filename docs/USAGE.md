# polypack usage guide

Worked examples for the common workflows. Install first:

```bash
pip install .            # CPU (JAX CPU wheel; everything works, search is slower)
pip install ".[cuda]"    # NVIDIA GPU search
pip install -e ".[test]" # development + tests
```

All commands below are console scripts installed with the package; each one
is also runnable as `python -m polypack.<module>` or, for campaign
compatibility, as `python polygon-packer/<module>.py`.

## 1. Certify a packing (the core loop)

Every claim starts and ends here. `certify` computes exact separating-axis
margins and converts any residual violation into a rigorous size bound:

```bash
polypack-validate paper/solutions/squ_in_oct_26.json
```

```
Worst pair separation:    +1.1e-16   (>= 0 is valid)
Worst containment margin: +3.2e-13   (>= 0 is valid)
Raw:       2.527094992258591
Certified: 2.527094992258591
Certification cost: 0.000e+00
```

From Python:

```python
import numpy as np
from polypack import load_solution, build_geometry, certify

sol, x, S = load_solution("paper/solutions/squ_in_oct_26.json")
rep = certify(np.asarray(x), S, build_geometry(4, 8), 4, 8)
print(rep["certified_side_length"])   # 2.527094992258591
```

The record semantics: a table entry `2.52711+` means the incumbent holds a
value in `[2.52711, 2.52712)`. You have a record only if your **certified**
value is strictly below the displayed floor — and on a live benchmark,
re-read the table the day you submit.

## 2. Search for a packing

```bash
# 21 triangles (3 sides) in a hexagon (6 sides), batches of 512, 40 rounds
polypack-search 21 3 6 --batch 512 --rounds 40

# warm-started from existing solutions, with an early-exit target
polypack-search 44 3 5 --seed-json "results/44_*.json" --target 3.52139
```

Each round: elites are perturbed by structured moves (vacancy teleports,
cluster tilts, shears), topped up with lattice/random immigrants, annealed
as one float32 GPU batch, and the best survivors are certified in float64.
Progress and `*_bh_top*.json` files land in the working directory.

Programmatic search on the same engine:

```python
import numpy as np, jax
from polypack import Engine

eng = Engine(8, 3, 6)                       # 8 triangles in a hexagon
anneal = eng.make_anneal(3e-8)
rng = np.random.default_rng(0)
xs = [np.column_stack((rng.uniform(-2, 2, (8, 2)),
                       rng.uniform(0, 2 * np.pi, 8))).ravel() for _ in range(64)]
Ss = np.full(64, eng.lowest_S * 3)
S_out, x_out = anneal(jax.random.PRNGKey(0),
                      np.asarray(xs, np.float32), np.asarray(Ss, np.float32))
i = int(np.argmin(S_out))
S64, x64, ok = eng.refine64(np.asarray(x_out)[i][None] * 1.001,
                            np.array([float(S_out[i]) * 1.001]))
print(float(S64[0]) * eng.ratio, bool(ok[0]))
```

## 3. Reconstruct a published packing from its image

Record pages publish images, not coordinates. Recover them:

```bash
# 26 hexagons in a pentagon whose table value is 7.04739+
polypack-reconstruct page_image.gif 26 6 5 7.04739 --out recon.json

# converge the reconstruction (it is deliberately 0.5% inflated)
polypack-validate recon.json --polish --squeeze
```

If the incumbent's published value was loose, the squeezed result certifies
below their displayed floor; if it was tight, you have documented a tie.
Calibration note (measured on 55 packings at site resolution in the
companion paper): reconstruction carries ~0.5 px mean center error, and
refinement is a *seeded perturbation search* — it lands in a nearby basin
within ~1e-4, not necessarily at the source value.

## 4. Propagate a strong packing downward

```bash
polypack-drop-one results/35_recon_refined.json --out-prefix results/34_drop
```

All `n` removal variants are squeezed as one batch. When your `(n+1)` basin
has better architecture than the incumbent at `n`, the improvement cascades
— the campaign's triangles-in-octagon 35 → 34 → 33 chain came from exactly
this command.

## 5. Hunt exact closed forms

```bash
polypack-exact paper/solutions/squ_in_tri_42.json
```

Extracts the active contact graph, solves the contact system at 160 digits
(quadratic convergence), then runs PSLQ on the refined side ratio. Accept a
relation **only** if its residual vanishes at ≥140 digits; the tool prints
both validated identifications (`3s²−36s+44 = 0`, i.e. `s = 6+8/√3`) and
rejected spurious candidates.

## 6. Track the live benchmark

```bash
polypack-scrape data/tables-today data/tables-yesterday
```

Scrapes all 24 tables to CSV and diffs against the prior scrape. Archive
every scrape you take: on a benchmark that moves daily, dated snapshots are
the only durable evidence of who held what when.

## 7. Render results

```bash
polypack-render solution.json -o packing.png            # quick look
polypack-render-webready solution.json sample.gif out.gif   # site-style GIF
```

## Reproducing the campaign's published claims

The repository's `paper/` directory contains the full evidence chain; two
commands re-derive it from the shipped coordinates:

```bash
python paper/collect_and_certify.py     # re-certify all 55 claim packings
python paper/certify_interval.py        # interval-arithmetic certificates
```
