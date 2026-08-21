# JOSS submission checklist

State as of 2026-08-17. JOSS review criteria mapped to where each is
satisfied, followed by the actions only the author can take.

## Review criteria → evidence

| JOSS criterion | Where satisfied |
|---|---|
| OSI-approved license file | `LICENSE` (GPL-3.0), declared in `pyproject.toml` |
| Substantial scholarly effort | 6-week campaign, 52 credited benchmark records, full evidence chain in repo (`LEDGER.md`, `ATTACK_MATRIX.md`, `paper/`) |
| Obvious research application | Benchmark records + companion preprint; statement of need in `joss/paper.md` |
| Installation instructions | `README.md` (Installation) + `docs/USAGE.md`; `pip install .` from `pyproject.toml` |
| Automated tests | `tests/` (pytest, 20 tests: known-value regressions, tamper detection, analytic SAT cases, engine smoke, PSLQ discipline, scraper parser, image round trip) |
| Continuous integration | `.github/workflows/ci.yml` (Linux, Python 3.11–3.13, CPU-only) |
| API/functionality documentation | `docs/API.md`, `docs/USAGE.md`, module docstrings |
| Example usage | `README.md` Quickstart + `docs/USAGE.md` worked examples |
| Community guidelines (contribute / report issues / seek support) | `CONTRIBUTING.md` |
| Software paper (summary, statement of need, refs) | `joss/paper.md` + `joss/paper.bib` |
| Version number | 1.0.0 (`pyproject.toml`, `polypack/__init__.py`, `CITATION.cff`, `CHANGELOG.md`) |
| Authorship | Sole author = campaign director; AI assistance disclosed in `joss/paper.md` acknowledgements (per JOSS AI policy) |

## Author actions before submitting

1. **ORCID**: add your ORCID iD in `joss/paper.md` (marked TODO). Create one
   at https://orcid.org if needed (takes minutes).
2. **Commit and push** all of this, and check the Actions tab shows the CI
   run green on GitHub's Linux runners.
3. **Tag a release**: `git tag v1.0.0 && git push --tags`, then create a
   GitHub Release from the tag (title "polypack 1.0.0", notes from
   `CHANGELOG.md`).
4. **Enable GitHub Issues** on the repository (Settings → Features) if not
   already on — CONTRIBUTING.md points there.
5. **arXiv link**: once the companion preprint is posted, add its arXiv ID to
   `joss/paper.md` ("companion research paper") and `README.md` (Citing).
   Not required by JOSS, but strengthens the research-application story.
6. **Submit** at https://joss.theoj.org/papers/new — repository URL, branch
   `main`, the paper is auto-detected at `joss/paper.md`. Suggested topic
   editor area: computational mathematics / applied math software.
7. **During review**: respond in the review issue; reviewers typically ask
   for small doc/test additions.
8. **At acceptance** (not before): archive the accepted release to Zenodo
   (link GitHub↔Zenodo, re-create the release or use "Publish" on the
   existing one), then post the Zenodo DOI in the review thread. JOSS
   requires the archive DOI at the end of review, not at submission.

## Paper compile check (optional, local)

JOSS builds `paper.md` automatically in the review thread (`@editorialbot
generate pdf`). To preview locally with Docker:

```bash
docker run --rm -v $PWD/joss:/data openjournals/inara -o pdf paper.md
```

## Scope note (if a reviewer asks)

The JOSS submission is the **software** (`polypack/` + tests + docs). The
research results it produced (the audit, the 52 records, the closed forms)
are the companion preprint's contribution — the standard software/research
split. `polygon-packer/polygon_packer.py` is the GPL-3.0 upstream this
package extends; the extension layers (engine, certifier, reconstruction,
exact solving, mechanisms, scraping — all of `polypack/`) are original to
this project, and the delineation is documented in `README.md` (Credits)
and `CHANGELOG.md`.
