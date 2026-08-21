# Polygon Packing Records

[![tests](https://github.com/AriPapowitz/polygon-packing-records/actions/workflows/ci.yml/badge.svg)](https://github.com/AriPapowitz/polygon-packing-records/actions/workflows/ci.yml)

**polypack** — a certified GPU toolkit for finding and verifying record
packings of unit regular polygons inside minimal regular containers — the
problems catalogued on
[Erich Friedman's Packing Center](https://erich-friedman.github.io/packing/).
This repository holds both the installable package (`polypack/`) and the
complete, reproducible evidence chain of the 2026 record campaign it ran.

During July 4–8, 2026 this toolkit claimed improved packings for **54 problems**
across 14 of the site's tables (triangles/squares/pentagons/hexagons/octagons in
various containers). **52 were credited** under the name *Aristotle Papowitz*
(verified against archived table scrapes), **41 still standing as of
August 16, 2026** — and every submitted packing re-certifies from its exact
coordinates (`paper/certification.csv`), with machine-rigorous
interval-arithmetic certificates in `paper/certification_interval.csv`. The
full reconciled ledger (every claim, margin, mechanism, and current status) is
in [LEDGER.md](LEDGER.md); the campaign's complete audit trail — 551 problems
examined, per-entry outcomes with evidence — is in
[ATTACK_MATRIX.md](ATTACK_MATRIX.md); the chronological campaign log — what
worked, what failed, and every negative result — is in
[CAMPAIGN.md](CAMPAIGN.md).

<p align="center">
  <img src="polygon-packer/results/night_trioct34.png" width="300"
       alt="34 unit triangles in the smallest known regular octagon">
  <br><em>34 unit triangles in an octagon of side 1.86454+ — found by dropping
  one triangle from a stronger 35-triangle packing and re-squeezing.</em>
</p>

## How records actually fall

Pure multi-start optimization converges to known optima but almost never finds
new ones. Every record this toolkit produced came from one of four *structured*
mechanisms, in descending yield:

1. **Reconstruct + squeeze** (`polypack-reconstruct`) — recover a competitor's
   coordinates from their published image (~0.1 px accuracy), then converge them
   in float64. Freshly-posted, mass-produced entries are often 1e-4 to 1e-3
   loose; squeezing them below their own displayed floor takes minutes.
2. **Structured basin-hopping** (`polypack-search`) — batched L-BFGS waves in
   float32 on GPU with an elite pool, vacancy/teleport/shear moves and lattice
   immigrants, certified on-device in float64. Finds genuinely new arrangements
   the incumbents missed.
3. **Drop-one propagation** (`polypack-drop-one`) — remove one shape from a strong
   (n+1)-packing and re-squeeze all n+1 removal variants as one batch. When your
   n+1 basin has better architecture than the incumbent's n, the improvement
   cascades downward through consecutive n.
4. **Exact solving** (`polypack-exact`) — contact graph → independent
   constraints via pivoted QR → min-norm Gauss–Newton in 160-digit
   arithmetic → PSLQ, with every candidate relation gated by residual
   validation. Identifies closed forms (or certifies their absence at a
   stated coefficient budget).

## Installation

```bash
pip install .            # CPU everywhere (search runs on the JAX CPU backend)
pip install ".[cuda]"    # NVIDIA GPU search (Linux; CUDA 12)
pip install -e ".[test]" # development install with pytest
```

Requires Python ≥ 3.11. Everything — including the whole test suite — runs
on CPU; the GPU only makes the search waves ~25× faster.

## Quickstart

```bash
# Search: 21 triangles (3 sides) in a hexagon (6 sides)
polypack-search 21 3 6 --batch 512 --rounds 40

# Certify a solution: float64 polish + container squeeze + exact margins
polypack-validate 21_3_in_6_bh_top1.json --polish --squeeze

# Reconstruct coordinates from a published packing image
polypack-reconstruct image.gif 26 6 5 7.04739 --out recon.json

# Best (n-1)-packing from an n-packing
polypack-drop-one recon.json --out-prefix 25_drop

# Hunt the exact closed form of a converged packing (squares, 160 digits + PSLQ)
polypack-exact paper/solutions/squ_in_tri_42.json

# Scrape all 24 record tables (and diff against a prior scrape)
polypack-scrape data/tables-today data/tables-yesterday
```

Campaign-era invocations (`python polygon-packer/validate_packing.py …`,
`import pack_core`) keep working through compatibility shims. Worked
examples: [docs/USAGE.md](docs/USAGE.md); full API:
[docs/API.md](docs/API.md).

Reproduce the campaign's published claims from the shipped coordinates:

```bash
python paper/collect_and_certify.py   # re-certifies all 55 claim packings
python paper/certify_interval.py      # interval-arithmetic certificates
```

Solution JSONs store the container circumradius, side ratio `s` (the number the
tables list), and per-shape `(x, y, angle)` placements — containers centered at
the origin with a vertex on the +x axis, shapes with circumradius
`1/(2·sin(π/m))` in unit-side units.

## Repository layout

| Path | What it is |
|---|---|
| `polypack/` | The installable package: engine, certifier, mechanisms, tooling |
| `tests/` | Pytest suite (known-value regressions against the published claims, tamper detection, engine + image round trips) |
| `docs/` | [API reference](docs/API.md) and [usage guide](docs/USAGE.md) |
| `joss/` | JOSS software-paper sources |
| `paper/` | The campaign's evidence chain: claim coordinates, certificates, ledgers, verification scripts |
| `polygon-packer/` | Campaign-era scripts + compatibility shims; includes the GPL-3.0 upstream `polygon_packer.py` |
| `polygon-packer/results/` | Solution JSONs, reconstructions, rendered images |
| `data/tables-*/` | Dated scrapes of all 24 record tables (CSV) |
| `CAMPAIGN.md` | Chronological campaign log: every experiment and verdict |
| `LEDGER.md`, `ATTACK_MATRIX.md` | The frozen record ledger and the 551-problem audit matrix |
| `RESEARCH.md`, `TARGETS.md`, `ATTACK.md` | Method survey and campaign planning notes |

Key modules in `polypack/` (each also a `polypack-*` console command):

| Module | Purpose |
|---|---|
| `pack_core` | Importable engine: penalties, natively batched L-BFGS, shrink-anneal, `refine64` (f64 polish/grow/squeeze), solution I/O |
| `packer_bh` | Structured basin-hopping CLI (elite pool, kicks, immigrants, seeding, target early-exit) |
| `validate_packing` | Exact separating-axis verification with certified dilation bounds — independent of the search stack |
| `reconstruct_gif` | Image → coordinates (color/gray/pastel fills, subpixel boundary fits) |
| `drop_one` | n → n−1 by batched removal + squeeze |
| `exact_solve` | 160-digit contact-system solving + residual-validated PSLQ closed-form detection |
| `build_lattice`, `scan_removals` | Exact lattice constructions and lattice-minus-holes scans |
| `render_packing`, `render_webready` | Quick renders, and site-style GIFs (style learned from a sample image) |
| `scrape_tables` | Scrape + diff the live record tables |

## Tests and documentation

```bash
pytest -q -m "not slow"   # fast suite: certification regressions, geometry, parsers
pytest -q                  # + engine smoke search and the image round trip (~4 min)
```

The reference values in the tests are the campaign's published claims: the
suite re-derives credited record values from the shipped coordinates to
1e-9. CI runs the suite on Linux (CPU-only) for Python 3.11–3.13.

## Contributing and support

Bug reports and questions: [GitHub issues](https://github.com/AriPapowitz/polygon-packing-records/issues).
Development setup, house rules (nothing counts until it certifies), and
contribution ideas: [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

If you use polypack, please cite it via [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository" button). A companion research paper on the
2026 campaign and benchmark audit is in preparation; a JOSS software paper
is under `joss/`.

## Practical lessons (the short version)

- **Truncation semantics matter**: a table value `2.90812+` means the incumbent
  holds a value in `[2.90812, 2.90813)`. You have a record only if your
  *certified* value is strictly below the displayed floor.
- **Verify against the live page the same day you submit.** Tables move daily;
  we lost four finished records by sitting on them for thirty hours while a
  competitor swept the same band.
- **Float32 sightings below a plateau are precision artifacts** — nothing is
  real until it survives float64 certification with positive exact margins.
- Negative results are documented in CAMPAIGN.md so you don't have to repeat
  them: the n=9–12 band of all tables is converged; plateau packings
  (`4+4√2`-type) resist rearrangement without shear-permissive moves; drop-one
  across ~400 seeds yielded exactly one (two-record) cascade.

## Credits

- Built on [Flamethr0wer's polygon-packer](https://github.com/Flamethr0wer/polygon-packer)
  (GPL-3.0), heavily extended: JAX GPU batching, float64 certification,
  basin-hopping structure, reconstruction, exact solving, and the campaign
  tooling around it.
- [Erich Friedman](https://erich-friedman.github.io/packing/) maintains the
  Packing Center and processes submissions with remarkable patience.
- The competitors whose relentless pace made this fun: Jonathan Viquerat,
  Bhavithran Ananthan, Jake Loyd, Haowei Lin, Aapo Lipponen, and the FICO
  Xpress team.

## License

GPL-3.0 — see [LICENSE](LICENSE). Inherited from the upstream engine and
applied to the whole toolkit.
