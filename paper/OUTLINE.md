# Paper outline — preprint first

Working doc for the arXiv preprint, to be adapted for journal submission afterwards.
Companion data: `LEDGER.md` / `paper/ledger.csv` (frozen 2026-08-16).

## Strategy

- **Preprint first** (arXiv), venue-agnostic LaTeX, then adapt. In a field where
  records fall weekly, the timestamp is worth more than the venue.
- **Categories**: math.OC (primary), cross-list math.MG, cs.CG.
- **Journal targets, in order**: J. of Global Optimization (computational packing
  lineage), Experimental Mathematics (if the exact-forms section grows), Computers
  & OR / J. of Heuristics (heuristic-search framing), Optimization Letters
  (compressed fallback). Not Math Programming tier — no optimality proofs.
- **The one-sentence thesis**: on a live, competitive packing benchmark, most 2026
  records are produced by fast-but-loose pipelines; a certified two-phase GPU
  pipeline can audit the entire benchmark, harvest the convergence gap at scale
  (47 of our 58 submissions), and localize exactly where further progress must
  come from structured construction rather than compute.
- **What this paper is NOT**: a trophy-case announcement. The 58 records are the
  *evidence*; the contribution is the audit + the certified pipeline + the
  structured mechanisms + the negative results. This framing is also the honest
  one: 47/58 came from re-converging other people's packings.

## Title candidates

1. *Auditing a live packing benchmark: certified GPU refinement and the anatomy
   of polygon-packing records* ← current favorite
2. *How packing records fall: a certified-computation audit of Erich Friedman's
   Packing Center*
3. *Improved packings of regular polygons in regular polygons via certified GPU
   search and structured construction*

Author: Aristotle Papowitz (single author). AI-assistance disclosure in §8/
acknowledgments per arXiv + venue policy.

## Abstract (draft skeleton, ≤200 words)

Context: 24 tables of polygon-in-polygon packing records, maintained since ~1998,
active 2025–26 arms race (NLP solvers, AlphaEvolve, mass-producing hobbyists).
Method: two-phase pipeline — batched float32 GPU search (natively batched L-BFGS,
shrink-anneal) → on-device float64 refinement (polish/grow/squeeze) → independent
exact-separating-axis certification (dilation ≤ 1e-12) → same-day live-table
verification. Four structured mechanisms on top: image reconstruction + squeeze,
seeded basin-hopping, drop-one propagation, high-precision contact solving + PSLQ.
Results: 54 problems claimed in five days (July 4–8, 2026; 58 submission
line-items) across 14 of 24 tables; **52 credited (scrape-verified), 41 still
standing 2026-08-16**; margins 3.9e-7 to 7.4e-3; every packing re-certified
from its submitted coordinates. Conservative-claims policy: the paper counts
only scrape-verifiable credits.
Audit findings: the 2026 record wave is systematically under-converged by
contributor-specific amounts (~1e-3 for two prolific setters, <1e-5 for others);
legacy entries (1997–2012) are converged ties; randomized multi-start cannot
reach rearrangement-class optima (n=21 control: 4,700+ restarts, 0 hits).
One 65-digit contact-system analysis with a provable no-closed-form certificate.

## Sections

### 1. Introduction (~2 pp)
- The benchmark: Erich Friedman's Packing Center, 24 categories, ~490 entries,
  28 years of history; submission = email + human verification; truncated display
  values. Why it matters: recent entrants include an industrial NLP-solver team
  (Berthold et al., arXiv:2605.04850, 22 records May 2026) and DeepMind's
  AlphaEvolve (hex-in-hex n=12 still standing against our search). It is the de
  facto shared testbed for polygon packing heuristics.
- The June–July 2026 arms race: records set daily; table velocity as a research
  condition (we lost 2 of 58 submissions purely to the table moving first).
- Contributions (C1–C5):
  C1 certified two-phase GPU pipeline (search f32 → refine f64 → certify exactly);
  C2 the audit: per-contributor convergence-gap demographics across the benchmark;
  C3 three structured record mechanisms with measured yields (reconstruct+squeeze,
     seeded/structured BH, drop-one propagation) + closed-form identification;
  C4 the ledger: 58 submissions / 54 cells / margins / attrition, all reproducible
     from archived scrapes + coordinates + verifier;
  C5 negative results that map the frontier: where compute stops working and
     construction must take over.

### 2. Problem, conventions, verification semantics (~1.5 pp)
- Pack n unit-side regular m-gons in the smallest regular M-gon; s = side ratio;
  the site's truncation convention (displayed `x+` ⇒ incumbent ∈ [x, x+1e-5)) and
  what "beating" means (certified value strictly below displayed floor).
- Feasibility = pairwise non-overlap + containment; our certificate: exact
  separating-axis margins in float64 + certified dilation bound ≤ 1e-12
  (validate_packing.py, independent of the search code).
- Precision discipline as a first-class topic: float32 acceptance produced false
  "records" with hidden 3e-4 violations; finite-difference squeezes stall 2e-4
  above basin bottoms. (This section is what most distinguishes us from the
  mass-producers — and it is measurable.)

### 3. The pipeline (~3 pp)  [C1]
- 3.1 Batched GPU search: penalty formulation; natively batched L-BFGS (two-loop
  recursion + Armijo with per-instance masks); batched shrink-anneal with kick
  rescue; 127 restarts/min on one RTX 4090 ≈ 25× a 32-thread CPU box; JAX,
  single-program-multiple-instance.
- 3.2 Two-phase precision: f32 exploration waves; top-survivor promotion to
  on-device f64 refine (polish → grow-to-feasible → squeeze).
- 3.3 Certification + live-table gate (same-day re-verification; the two DOA case
  studies as motivation).
- Figure F1: pipeline diagram. Table T2 seed: throughput + false-record rate of
  f32-only acceptance.

### 4. Structured mechanisms (~4 pp)  [C3]
- 4.1 Reconstruct + squeeze (the audit instrument): published GIF/PNG →
  coordinates at ~0.1–0.15 px (color/gray/pastel fills, subpixel boundary fits) →
  f64 convergence. This is how 47/58 records fell. Framing: measuring the
  *convergence gap* of live entries; incumbents credited as the source
  construction in every case.
- 4.2 Structured basin-hopping: elite pool; vacancy/teleport/tilt-cluster/shear
  moves; lattice±defect immigrants; penalty ramp. Case study: squ_in_oct 36 —
  trivial-37-lattice-minus-one seed → vacancy rearrangement 2.92870 < 10−5√2
  (the "five-minute record" story told honestly, incl. the lesson about trivial
  n+1 dominance checks).
- 4.3 Drop-one propagation: batch removal + squeeze; better-basin cascade
  tri_in_oct 35 → 34 (−7.2e-3) → 33 (−4.4e-3), halting at converged 32; measured
  yield: ~400 seeds → exactly one (two-record) cascade.
- 4.4 Exact solving: contact graph → independent constraints (pivoted QR) →
  min-norm Gauss–Newton in 80–160-digit arithmetic (quadratic convergence
  1e-8 → 1e-80) → PSLQ. Results: squ_in_tri 42/43 agree with 6+8/√3 and 5+10/√3
  to ~4e-9 (settle exactly pre-submission — task V2); squ_in_oct 26 at 65 digits
  with **no minimal polynomial ≤ deg 8 (coeffs ≤ 1e8)** — a no-closed-form
  certificate; structure: 4 rattlers, 8 floppy modes, 59 independent contacts.
  Figure F6: contact graph with rattlers highlighted.

### 5. The campaign as experiment (~4 pp)  [C2, C4]
- 5.1 The ledger: Table T1 = condensed LEDGER.md (58 submissions, 54 cells,
  margins, mechanisms, status). Full table + coordinates in supplementary.
- 5.2 Convergence demographics: margin-vs-prior-holder distributions (harvest
  margins cluster ~1e-3–4e-3 for two prolific 2026 setters; <2e-5 for two others;
  legacy 1997–2012 entries and all closed-form entries: exact ties). Figure F3.
  Respectful framing: convergence gap of a pipeline, not competence of a person.
- 5.3 Negative results [C5]:
  (a) n=9–12 band across all tables: 69 targets, 0 records — fully converged era;
  (b) legacy tri_in_hex 8/16/17/18: reproduce record values to 1e-9 — true ties;
  (c) tri_in_hex n=21 control: 640 CPU + 4,096 GPU restarts + 119 structured
      scans never reach the known-better rearrangement basin — randomized
      multi-start provably (empirically) insufficient; frontier is
      construction-limited;
  (d) plateau rigidity (s=2 tri-in-hex 22–24; 4+4√2 oct-in-squ; staircase holes);
  (e) drop-one sweep yield (400 → 1).
- 5.4 Benchmark velocity: attrition curve (52 credited → 49 → 41 over 6 weeks);
  the two DOA submissions; the four sniped-while-staged records; median record
  lifetime on fresh tails. Figure F4. Implication: date-stamped claims +
  archived-scrape provenance are methodologically necessary (and what we do).

### 6. Related work (~1.5 pp)
- Circle packing record lineage (packomania-era; Graham/Lubachevsky;
  Addis/Locatelli/Schoen; Markót–Csendes interval *proofs* — contrast: we certify
  feasibility, not optimality).
- Polygon packing: Friedman's dynamic survey (squares in squares, EJC);
  Gensane–Ryckelynck improved square packings (the genre precedent);
  Kallus/Toth-style rigidity where relevant.
- 2025–26 entrants: Berthold et al. NLP-solver records (incl. their symmetry
  ablation), AlphaEvolve / ImprovEvolve / LLM-evolved constructors, Viquerat's
  learning-based packing. Sources gathered in RESEARCH.md; needs real bib pass
  (task W5).
- GPU batched local optimization (batched L-BFGS precedents).

### 7. Discussion & limitations (~1 p)
- Upper bounds only; no optimality proofs (contrast Markót–Csendes; note
  exact_solve gives *local* rigidity structure, not global optimality).
- One benchmark family (regular-in-regular); generalization plausible but unshown.
- What a "healthy" live benchmark would need: certification requirements,
  full-precision coordinate publication, snapshot archives. (Constructive, not
  preachy — Erich's site already does human verification remarkably well.)
- The constructor bottleneck: compute reproduces known optima to n≈22 and falls
  structurally behind by n≈28; the record-setters who matter are winning on
  construction. Points at ranked future work (SCIP/Farkas warm starts,
  LLM-evolved move programs).

### 8. Reproducibility, data, disclosure (~0.5 p)
- Repo (github.com/AriPapowitz/polygon-packing-records): engine, verifier,
  ledger builder, archived scrapes, all coordinates. One-command re-certification
  of every claimed packing (task V1 builds this).
- AI disclosure: campaign orchestration and code substantially AI-assisted
  (Claude); every packing machine-generated and independently certified; all
  submissions human-reviewed by the site maintainer. Wording per arXiv/venue
  policy at submission time.
- Acknowledgments: Erich Friedman (maintenance + patient processing); named
  competitors whose entries seeded harvests (they are cited in-table).

## Figures & tables

- F1 pipeline diagram (f32 wave → f64 refine → certify → live gate → submit)
- F2 gallery: squ_in_oct 36 rearrangement; tri_in_oct 34 drop-one; squ_in_tri 42
  closed form; one harvest before/after overlay (recon dots on incumbent GIF)
- F3 margin-by-prior-holder strip plot (from ledger.csv)
- F4 attrition/velocity: standing count over the 6 scrapes + churn context
- F5 search-yield vs n (ties/records/misses per band: n≤8 blitz, 9–12, legacy,
  21–24 plateaus, ≥28 fresh tails)
- F6 squ_in_oct 26 contact graph, rattlers/floppy modes marked
- T1 condensed ledger (14 categories × what fell, margins, status)
- T2 mechanism yields (targets → ties → records → still standing, per mechanism)
- T3 negative results summary
- Supplementary: full ledger CSV, all 54 coordinate sets, certification
  transcript, scrape archive listing.

## Pre-submission verification checklist (do before arXiv)

- V1 **Batch re-certification**: ✅ DONE 2026-08-16 — `paper/collect_and_certify.py`
  rebuilds all 54 claims (+ the n8 tie) from the sent artifacts and re-certifies:
  55/55 reproduce claimed values to ≤1e-9 (`paper/certification.csv`,
  `paper/solutions/`). Re-run the week of posting.
- V2 **Settle the closed forms**: exact_solve on squ_in_tri 41/42/43 (squares —
  supported); report "= 6+8/√3 (exact-solve to N digits)" or "agrees to 4e-9,
  form unproven" — whichever the computation supports. Also re-verify the
  10−5√2 trivial-lattice statement for squ_in_oct 36/37.
- V3 **Freshest scrape** the week of posting; update LEDGER "standing" count and
  the attrition figure (the paper claims a dated snapshot, not eternity).
- V4 **Fact-check every campaign number** quoted in the paper against logs:
  127 restarts/min, 25×, 4,700+ restarts at n=21, 69/69, ~400 drop-one seeds,
  0.1–0.15 px reconstruction accuracy, cloud spend (~$60? — pin down from
  invoices/logs before quoting).
- V5 **Name spellings** of all credited competitors from the live site; get
  Berthold et al. author list + correct arXiv number from the actual paper.
- V6 **Site-owner courtesy**: short email to Erich Friedman before posting —
  the paper audits his benchmark; he processed 58 submissions; a heads-up is
  both polite and prudent. (Not permission — courtesy.)

## Writing plan

- W1 §3 pipeline + §4 mechanisms (the parts that exist in code — write from
  source, cite file names)                                    [~3 days]
- W2 §5 results: generate F3/F4/F5 + T1/T2 from ledger.csv + logs   [~2 days]
- W3 §2 + §7 + §6 related-work skeleton                        [~2 days]
- W4 §1 intro + abstract last                                  [~1 day]
- W5 bib pass: real citations (WebSearch/Scholar), Berthold/AlphaEvolve/
  Gensane/Markót etc.                                          [~1 day]
- W6 red-team pass: adversarial referee review of the draft (attack overclaims,
  check every number), then fix                                [~1 day]
- W7 LaTeX/arXiv packaging, ancillary files, post              [~1 day]
- Then: JOSS packaging (separate track, see below) and journal adaptation.

## The JOSS split (phase 3, separate submission)

JOSS paper = the *software* (pack_core batched-L-BFGS engine, validate_packing
certifier, reconstruct_gif, exact_solve, scrape/diff tooling) with a statement of
need; the preprint is the research that used it — the standard, accepted split.
Repo needs before JOSS: tests (validator round-trips, known-value regression on
3–4 published packings), pip-installable packaging, API docs, CONTRIBUTING,
tagged release + Zenodo DOI. Original layers must be clearly delineated from the
GPL-3.0 upstream (Flamethr0wer/polygon-packer) it extends. ~1–2 weeks of work.
