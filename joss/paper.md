---
title: 'polypack: certified search and verification for packings of regular polygons in regular polygons'
tags:
  - Python
  - computational geometry
  - packing problems
  - global optimization
  - certified computation
  - GPU
  - experimental mathematics
authors:
  - name: Aristotle Papowitz
    # TODO before submission: add ORCID, e.g.  orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 17 August 2026
bibliography: paper.bib
---

# Summary

How many unit equilateral triangles fit in a regular hexagon of a given
side? For over twenty-five years, Erich Friedman's *Packing Center*
[@friedman-site; @friedman-survey] has curated best-known answers across 24
tables of regular-polygon-in-regular-polygon problems. In 2025--26 the site
became a live arena: DeepMind's AlphaEvolve used it as a public testbed
[@alphaevolve], an industrial nonlinear-programming team swept dozens of
entries [@berthold-oob; @berthold-hex], evolutionary hybrids followed
[@improvevolve], and prolific individual contributors pushed the tables
daily.

`polypack` is the Python toolkit built for competing on — and auditing —
this benchmark. It provides (1) a batched search engine: a penalty
formulation minimized by a natively batched L-BFGS [@lbfgs] under JAX
[@jax], running thousands of simultaneous restarts in float32 on a consumer
GPU with on-device float64 refinement (polish, grow-to-feasible, squeeze);
(2) an independent certifier that computes exact separating-axis margins
and converts any residual violation into a rigorous dilation bound, so
every claim carries a *provably valid* container size; (3) reconstruction
of competitor packings from published images at subpixel accuracy; (4)
structured record mechanisms (basin-hopping with vacancy/tilt/shear moves
and lattice seeding; batched drop-one propagation from $(n{+}1)$- to
$n$-packings); (5) exact contact-manifold solving in 160-digit arithmetic
[@mpmath] with PSLQ integer-relation detection [@pslq], gated by residual
validation; and (6) scrape/diff tooling that archives the moving benchmark.
The numerical core uses NumPy [@numpy], SciPy [@scipy], and numba [@numba],
and extends an open-source packer [@vallejo-packer] (GPL-3.0).

# Statement of need

Packing records occupy a peculiar evidentiary niche: they are upper bounds
whose only proof is the packing itself, published on leaderboards that are
increasingly cited as evidence of solver and AI capability. The circle
packing community solved the governance problem long ago — Packomania has
published full-precision coordinates for decades [@specht], with a lineage
of documented heuristic sweeps [@nurmela; @graham-lubachevsky; @amore] and
even interval-arithmetic optimality proofs [@markot-csendes]. The polygon
tables have printed precedents [@gensane; @kampas] but no shared tooling:
entries are images plus truncated decimals, submissions are verified by one
maintainer, and none of the 2025--26 wave's high-volume pipelines published
a verifier.

That gap has a measurable cost. A float32 search stage accepts "packings"
with hidden violations around $3\times10^{-4}$ — large enough to fake a
record — and finite-difference refinement stalls well above true local
optima. `polypack`'s two-phase discipline (fast float32 exploration,
float64 refinement, then independent exact certification, with
interval-arithmetic re-certification available for published claims) exists
because both failure modes produced false records in our own early runs.

The toolkit's capabilities were demonstrated in a five-day campaign (July
2026) on the live benchmark: certified improvements for 54 problems across
14 of the 24 tables, 52 credited by the site, 41 still standing at an
August 2026 freeze; three of the new records were identified as exact
algebraic values ($6+8/\sqrt{3}$ twice, $5+10/\sqrt{3}$) by the contact
solver, with every candidate relation subjected to residual validation. The
same instruments turned inward quantify their own limits: rendering our own
packings at site resolution and re-harvesting them shows reconstruction is
a seeded perturbation search, and found further certified improvements to
our own records at the $10^{-4}$ scale. A companion research paper (in
preparation) reports the full audit; this software is the infrastructure
that makes such claims checkable by anyone — the repository ships every
claimed coordinate set, dated table snapshots, and one-command
re-certification.

`polypack` is intended for three audiences: packing enthusiasts who want
their submissions certified rather than hoped-for; optimization and
metaheuristics researchers who need a reproducible, GPU-ready baseline and
a verifier for a benchmark family with genuine competitive pressure; and
maintainers of record leaderboards who want machine-checkable evidence
standards.

# Acknowledgements

Erich Friedman has maintained the Packing Center, and verified a great many
submissions including ours, with patience and care. The toolkit extends
Ignacio Vallejo's open-source `polygon-packer` [@vallejo-packer]. The
software was developed with substantial AI assistance (Claude, Anthropic)
under the author's direction; every packing it produces is independently
certified by the verification layer described above, and all claimed
records were additionally verified by the site's maintainer before listing.

# References
