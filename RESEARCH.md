# How Packing Records Actually Get Set — Research Synthesis (2026-07-04)

*Deep-research run: 5 search angles, 21 primary sources fetched, 102 claims extracted.
Verification pass was cut short by session limits — 4 claims formally confirmed (3-0
votes), the rest are direct quotes from primary arXiv/GitHub sources and should be
treated as reliable-but-unverified. Raw claim data: workflow `wf_71fca0ae-50a` journal.*

## The headline

Our diagnosis from the campaign was correct and is now confirmed by the literature:
**every current record-setting method is "structured construction + local refine",
never blind multi-start.** Three method families are actively winning on Friedman's
tables in 2025–2026, and two of them are fully reproducible by us:

1. **Off-the-shelf global NLP solvers** (Berthold et al., FICO/ZIB, arXiv:2605.04850 +
   arXiv:2601.05943) — 22 new best-known solutions in one paper.
2. **LLM-evolved basin-hopping** (AlphaEvolve arXiv:2506.13131; ImprovEvolve
   arXiv:2602.10233) — beat AlphaEvolve's own records with a *seeded* evolve-the-moves
   loop whose local optimizer is nearly identical to our `packer_gpu.py`.
3. **Structured move operators from the circle-packing literature** (vacancy search,
   billiard compression, lattice seeding) — decades of published evidence that these
   beat restarts on mature tables.

## Competitor intel

| Who | Method (evidence) |
|---|---|
| **Berthold / FICO-ZIB team** | Farkas-lemma separating-hyperplane NLP (12 multipliers per hexagon pair), solved by *unmodified* SCIP 10 / Xpress 9.8: ~5,000s multistart local NLP seeds a ~10,000s spatial branch-and-bound, 5 solver setups, then arbitrary-precision polishing (SAT-based uniform rescale, 1e-14 margin). 22 new records: displaced Cantrell (tri-in-squ n=12: 3.13403 vs 3.13802), Morandi, Friedman, Hirsh, AlphaEvolve. Hit tri-in-pen n=6,10–14 (n=14: 3.07677 vs 3.13711!), squ-in-pen n=9–13, tri-in-hex n=10,12,13, hex-in-hex n=11,12,14–17,23. **Their ablation: a redundant pairwise center-distance constraint was the single most impactful formulation choice; symmetry-breaking constraints made 539/750 instances WORSE (median −2.178%).** |
| **AlphaEvolve** (DeepMind, May 2025) | Did NOT evolve packings — evolved *heuristic search programs* (Gemini as mutation operator), each given ~1,000s and warm-started from the previous best construction. Set hex-in-hex n=11 (3.930092, tilting each hexagon at varying angles vs. flush) AND n=12 (3.9419123). Across 50+ math problems: ~75% rediscovered SOTA, ~20% surpassed. |
| **ImprovEvolve** (arXiv:2602.10233) | Beat AlphaEvolve on hex-in-hex n=11,12,15,16 (+human-edit variant: 14,17,23). Architecture: LLM-evolves a Python class with `generate_config` (structured feasible init) / `improve` (local optimizer) / `perturb` (sigma-scaled structured shake), run as scheduled basin-hopping **seeded with the best known packing**. Its evolved `improve` is penalty L-BFGS-B with exact JIT gradients and a staged penalty-weight ramp 1e3→1e9 with shrinking margins — *essentially our engine*. The differentiator is the moves and the seeding, not the optimizer. |
| **Viquerat** (~177 records) | Public method: PBO, policy-based optimization — a single-step deep-RL / evolution-strategy black-box optimizer (arXiv:2104.06175, code: github.com/jviquerat/pbo). CONFIRMED 3-0. But his own page says PBO only *matched* literature up to 26 DOF — it does not explain 177 records; his real pipeline is undisclosed. CONFIRMED 3-0. |
| **Jake Loyd** (~190 records) | No public method found. June 2026 wave timing is consistent with applying the Berthold paper's (public, May 2026) recipe or similar at scale. |
| **Specht** (Packomania) | MBS — Modified Billiard Simulation (Szabó & Specht 2005): random points "blown up" with collision handling + NEAR/FAR neighbor pruning; seeded with hexagonal/lattice initial configurations chosen by relating n to expected structure. Code + packings to n=300 were downloadable from packomania.com. |
| **Graham/Lubachevsky** (historic disk records) | Lubachevsky–Stillinger event-driven billiard compression (elastic disks, radii grow until jam) — set disk-in-triangle n=22–34 in one sweep (arXiv:math/0406252). The 2004 disk-in-square wave was a hybrid: cheap greedy compression phase → LS as high-precision finisher (arXiv:math/0405310). Also: Boll–Donovan–Graham–Lubachevsky 2000 set records (n=32,37,48,50) with a *trivially simple* compass-direction perturbation + shrinking step size — an 8th-grader improved records with it. |
| **Amore** (arXiv:2212.12287) | Reformulates as unconstrained minimization of a repulsive potential (container handled by parametrization); "pour, settle, shake" physical construction; reports basin-hopping added *no measurable benefit* for circles in regular polygons. Mixed evidence — treat basin-hopping as necessary-but-not-sufficient without good moves. |
| **Kampas/Pintér/Castillo** (arXiv:1901.07056) | NLP with embedded Lagrange multipliers + plain randomized multi-start (one local solve per start) — i.e. the strategy we already proved insufficient. Confirms our negative result is representative of naive approaches. |

## Ranked experiment list (payoff ÷ effort)

### 1. Structured basin-hopping on the GPU engine — DO FIRST
Replace random kicks in `packer_gpu.py` with a portfolio of *structured* moves,
seeded from best-known/lattice configurations (not random starts):
- **vacancy/insertion moves** (Huang & Ye GVS improved 41/200 mature circle-in-square
  records with exactly this: find the emptiest region, teleport the worst-fitting item
  into it, re-relax)
- **lattice ± defect seeds** (we already have `build_lattice.py` / `scan_removals.py`)
- **tilt-cluster moves** (AlphaEvolve's winning hex motif was uniform-tilt-per-hexagon;
  perturb a *contiguous cluster's* angles coherently, not iid noise)
- **row/column shear shifts**
- **ImprovEvolve's penalty schedule**: staged penalty-weight ramp 1e3→1e9 with
  shrinking margins instead of our fixed-tolerance accept (trivial to add).
Effort: LOW (all pieces exist). Evidence: ImprovEvolve beat AlphaEvolve with our
optimizer + these ingredients; GVS is direct proof structured moves beat restarts.

### 2. Farkas-NLP + SCIP branch-and-bound, warm-started by our GPU engine
Implement Berthold's formulation (Farkas separating-plane multipliers per pair
+ the redundant center-distance cut; NO symmetry-breaking constraints). SCIP is free.
Their protocol: multistart local → spatial B&B ~10ks → exact-arithmetic polish.
Hybrid edge nobody has published: warm-start the B&B with our GPU top-64 survivors.
Effort: MEDIUM (model writing + SCIP setup). Evidence: 22 records in one paper.
Caveat: the easy pickings from their own 750-instance sweep are gone — target n's
they didn't run, or feed them better incumbents than their 5,000s multistart found.

### 3. LLM-evolved move programs around our evaluator (ImprovEvolve recipe)
Point an open AlphaEvolve clone at our JAX penalty evaluator with the three-slot
API (`generate_config` / `improve` / `perturb`), always seeding with current
best-known. Open tooling, all Apache-2.0:
- **ShinkaEvolve** (SakanaAI) — claims SOTA circle packing in only ~150 program
  evaluations (parent sampling + novelty rejection + UCB over LLM ensemble)
- **OpenEvolve** (algorithmicsuperintelligence/openevolve) — MAP-Elites + islands
- **CodeEvolve** (inter-co/science-codeevolve)
Effort: MEDIUM. Evidence: this exact pattern set the current hex-in-hex records.

### 4. Billiard/compression dynamics as a *diversifier* (not replacement)
LS-style event-driven or soft-sphere compression reaches jammed configurations
gradient flow doesn't. For polygons, a cheap approximation: repulsive-potential
dynamics (Amore-style) or grow-shapes-until-jam inside our JAX sim, then hand
survivors to L-BFGS refine. Reference C++ LS implementation:
github.com/VasiliBaranov/packing-generation. Effort: MEDIUM-HIGH. Evidence: every
historic circle record wave (Graham/Lubachevsky, Specht MBS) used compression physics.

### 5. De-prioritized / negative results (save the effort)
- **Symmetry-restricted search**: Berthold's ablation — symmetry-breaking constraints
  degraded 539/750 instances. As *hard constraints during search*: don't. (Symmetric
  *seeding* is unaffected and still fine — Specht used lattice seeds.)
- **PBO / plain CMA-ES**: publicly only matches literature to ~26 DOF.
- **More random restarts at any scale**: settled, by us and by Kampas et al.
- **Plain basin-hopping with iid noise kicks**: Amore reports no benefit; the value
  is entirely in structured moves + seeding.

## Immediate housekeeping before any new attack

**TARGETS.md is stale.** Berthold's paper (public since May 2026) already took:
tri-in-pen n=6,10–14 · squ-in-pen n=9–13 · tri-in-hex n=10,12,13 · tri-in-squ n=12 ·
hex-in-hex n=11,12,14–17,23. Several of our Tier-1 rows point at values that are
likely already displaced (and the June-2026 Loyd/Viquerat wave moved more). Re-scrape
the live tables and re-rank targets before spending GPU time. Also check whether
Berthold's 22 solutions are on Friedman's page yet — any that aren't are *published*
values (can't be claimed by us), so those entries are dead targets either way.

## Sources (primary)

- arXiv:2605.04850 — Berthold et al., Out-of-the-Box Global Optimization for Packing (the 22-record paper)
- arXiv:2601.05943 — same team, hexagon NLP + AlphaEvolve comparison
- arXiv:2506.13131 — AlphaEvolve white paper (+ github.com/google-deepmind/alphaevolve_results)
- arXiv:2602.10233 — ImprovEvolve
- arXiv:2104.06175 + github.com/jviquerat/pbo — Viquerat PBO [CONFIRMED 3-0]
- jviquerat.github.io — Viquerat packing disclosure [CONFIRMED 3-0]
- arXiv:math/0406252, math/0405310 — Graham/Lubachevsky record waves
- inf.u-szeged.hu/~pszabo/Pub/45survey.pdf — Szabó circle-in-square survey (Specht MBS history)
- combinatorics.org ds7 — Friedman's own squares-in-squares survey (why circle methods don't transfer to rotating shapes)
- Huang & Ye — Greedy Vacancy Search (ResearchGate 220059755)
- arXiv:2212.12287 — Amore, circles in regular polygons
- arXiv:1901.07056 — Kampas/Pintér/Castillo
- github.com/SakanaAI/ShinkaEvolve · github.com/algorithmicsuperintelligence/openevolve · github.com/inter-co/science-codeevolve
- github.com/VasiliBaranov/packing-generation — LS reference implementation
