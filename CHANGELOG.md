# Changelog

## [1.0.0] — 2026-08-17

First packaged release, prepared for JOSS submission.

### Added
- `polypack/` package: the maintained library and CLIs, pip-installable
  (`pip install .`), with console entry points (`polypack-search`,
  `polypack-validate`, `polypack-reconstruct`, `polypack-drop-one`,
  `polypack-exact`, `polypack-lattice`, `polypack-scan-removals`,
  `polypack-render`, `polypack-render-webready`, `polypack-scrape`).
- Test suite (`tests/`, pytest): known-value regressions against the
  published claim coordinates, certifier tamper detection, analytic
  separating-axis cases, engine smoke search, refine-reconvergence,
  contact-extraction and residual-validated-PSLQ checks, scraper parser
  tests, and a full render→reconstruct→refine round trip.
- GitHub Actions CI (Linux, Python 3.11–3.13, CPU-only).
- Documentation: `docs/API.md`, `docs/USAGE.md`; community files
  (`CONTRIBUTING.md`, `CITATION.cff`); JOSS paper sources under `joss/`.

### Changed
- `packer_bh`, `drop_one`, `reconstruct_gif`, `render_webready`,
  `scrape_tables`, `exact_solve` gained proper `main(argv=None)` entry
  points (previously top-level script bodies).
- `scrape_tables.fetch` now uses urllib with a PowerShell fallback
  (previously PowerShell-only, i.e., Windows-only).
- `exact_solve` writes its report next to the input JSON (previously into a
  hard-coded `results/` relative to the working directory).

### Compatibility
- Every campaign-era flat module in `polygon-packer/` is now a shim that
  forwards to `polypack.*`; existing commands
  (`python polygon-packer/validate_packing.py …`) and imports
  (`import pack_core` with `polygon-packer/` on `sys.path`) work unchanged,
  as do the reproduction scripts in `paper/`.
- `polygon-packer/polygon_packer.py` (the GPL-3.0 upstream engine by
  I. Vallejo), `packer_gpu.py`, `packer_jax.py`, and `closed_form_sweep.py`
  are campaign-era artifacts left untouched.
