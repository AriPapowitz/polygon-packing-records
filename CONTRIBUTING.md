# Contributing to polypack

Thanks for your interest! This repository holds both a maintained Python
package (`polypack/`) and the frozen artifacts of the 2026 record campaign
(`paper/`, `data/`, the campaign logs). Contributions target the package;
the campaign artifacts are an archival record and stay as they are.

## Getting help / reporting problems

- **Bugs and questions**: open a
  [GitHub issue](https://github.com/AriPapowitz/polygon-packing-records/issues).
  For a suspected wrong certification, attach the solution JSON — it is the
  complete reproducer.
- **Security or data concerns** (e.g., something in the archived scrapes that
  should not be there): open an issue or email the author.

## Development setup

```bash
git clone https://github.com/AriPapowitz/polygon-packing-records
cd polygon-packing-records
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest -q -m "not slow"     # fast suite (~seconds after the first numba build)
pytest -q                    # full suite incl. engine + image round trip (~4 min)
```

No GPU is needed for development: JAX falls back to CPU and the whole test
suite runs there. For GPU search, install the `cuda` extra
(`pip install -e ".[cuda]"`).

## What makes a good contribution

- **Fixes and portability improvements** to the package modules.
- **New verified mechanisms** (constructors, moves, refinement strategies) —
  with a test that certifies their output via `polypack.certify`.
- **Test cases**: additional known-value regressions are always welcome; the
  published solutions in `paper/solutions/` are the reference corpus.
- **Documentation**: `docs/API.md` and `docs/USAGE.md`.

House rules, learned the hard way during the campaign and enforced by review:

1. **Nothing counts until it certifies.** Any code path that produces a
   packing must be demonstrated through `validate_packing.certify` (exact
   separating-axis margins + rigorous dilation bound). Float32 output is
   never evidence.
2. **Keep the certifier independent.** `validate_packing` must not import
   from the search stack (`pack_core`, `packer_bh`); the audit's credibility
   rests on that separation.
3. **Don't break the campaign paths.** The `polygon-packer/*.py` shims and
   `paper/*.py` scripts must keep running — `pytest` plus
   `python paper/collect_and_certify.py` is the compatibility check.

## Style

Match the existing code: plain numpy/JAX, small modules, docstrings that
state conventions (container vertex at angle 0, circumradius `S`, side ratio
`s = S·sin(π/nsc)/sin(π/nsi)`), comments only where the code cannot say it.

## License

GPL-3.0 (inherited from the upstream engine this package extends). By
contributing you agree to license your work under the same terms.
