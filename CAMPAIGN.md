# Packing Records Campaign — July 2–3, 2026

## Layout (reorganized 2026-07-04)

- Root: `CAMPAIGN.md` (this file) · `TARGETS.md` (re-scraped 2026-07-04) · `RESEARCH.md` (method synthesis) · `data/tables-2026-07-04/` (scraped record tables, CSV) · `submissions/` (sent artifacts)
- `polygon-packer/`: the tools (top level) + `results/tri_in_hex/` (solution JSONs/PNGs) + `logs/` (run logs) + `runpod/` (cloud-GPU playbook & network utils)
- Note: `packer_gpu.py`/`packer_jax.py` write output JSONs to the cwd — run them from `results/tri_in_hex/` (or move outputs after) to keep the top level clean.

One overnight campaign: from `git clone` to a submitted record attempt, a GPU
search farm, and a definitive map of where records in this field actually come
from. Credit target: Aristotle Papowitz. Total cloud spend: ~$3.50.

## What was built (all in `polygon-packer/`)

| Tool | What it does |
|---|---|
| `polygon_packer.py` (fixed) | Upstream tool + our fixes: `=`/`==` SyntaxErrors, the `lowest_S=√N` infinite-loop bug (area-ratio bound + shrink guard), JSON coordinate output |
| `packer_jax.py` | JAX port with exact autodiff gradients — ~30× faster wall-clock than the original using all cores |
| `packer_gpu.py` | **The engine.** Fully batched GPU search: natively-batched L-BFGS (two-loop + Armijo, per-instance masks), batched shrink-anneal with kick rescue, and two-phase precision — float32 exploration + on-device float64 refine (polish → grow-to-feasible → squeeze) of each wave's top survivors. 127 restarts/min on an RTX 4090 (~25× a 32-thread CPU box) |
| `validate_packing.py` | Independent certification: exact SAT margins, polish, squeeze, and a rigorous certified-size bound. Final gate before any claim |
| `build_lattice.py`, `scan_removals.py` | Exact 24-triangle hexagon tiling ± hole variants; structured squeeze scans |
| `render_packing.py` | Submission-grade images from solution JSONs |
| `chunked_get.py`, `mss_get.py`, `RUNPOD.md` | The hostile-network playbook: MSS-clamped parallel range downloads (533 B/s → 1.6 MB/s on a broken RunPod host, ~3000×) |
| `pack_core.py` (2026-07-04) | packer_gpu's engine factored into an importable library (penalty/L-BFGS/anneal/refine64, parameterized) |
| `packer_bh.py` (2026-07-04) | **Structured basin-hopping**: elite pool + vacancy/tilt/shear/shake moves + lattice immigrants. Smoke test: re-finds n=8 tri-in-hex record basin in 3 CPU rounds |
| `drop_one.py` (2026-07-04) | n→n−1 monotonicity constructor (batch removal scan + squeeze) |
| `reconstruct_gif.py` (2026-07-04) | Site GIF → coordinates (~0.15 px accuracy); n=45 tri-in-pen reconstructed. See `ATTACK.md` for the 4-GPU runbook |

## Experiments and verdicts

| Experiment | Result | Verdict |
|---|---|---|
| tri-in-hex n=8 (record 1.356+, Cantrell 2012) | Found 1.356597399687052 certified; **submitted; Erich confirmed NOT a record** | Cantrell's basin, fully converged — true tie |
| n=16/17/18 (records 9/5, 11/6, 17/9, Morandi 2008) | Refined to the exact record values +1e-9 | True ties; records are locally optimal |
| n=21 control (Watson's 1.99878 known to exist) | 640 CPU + 4,096 GPU restarts + 119 lattice-hole squeeze scans: never seen | Random multi-start cannot reach rearrangement-class basins |
| n=22/23 plateau (2.000 trivial) | All approaches bottom at ≥2.000; f32 "sub-2.000" sightings were violations in disguise | Plateau stands; would need structured search |
| Fresh tails: n=28 (Loyd 2.32251+, June 2026) | One GPU wave: best 2.33299 — miss by 0.0105 | June bulk records are structurally better, not soft |
| n=32 wave | Killed at shutdown before finishing | No data |

## The three lessons

1. **Precision discipline is everything.** float32 acceptance produced a false
   "record" (violations ~3×10⁻⁴ hiding below tolerance); finite-difference
   squeezes stall ~2×10⁻⁴ above true basin bottoms. Only the two-phase
   f32-search + f64-refine + independent certification pipeline produces
   numbers you can put your name on.
2. **The incumbents are fully converged.** Every record we could test — from
   2008 Morandi fractions to 2012 Cantrell numerics — sits at the exact bottom
   of its basin. There is no free slack anywhere we looked.
3. **Records come from better constructions, not more compute.** GPU-scale
   random search reproduces known optima up to n≈22 and falls structurally
   behind by n≈28. The active record-setters (Loyd ~190 records, Viquerat
   ~177, AlphaEvolve) are winning with structured search strategies.

## The next chapter (if/when resumed)

**2026-07-04: deep-research pass done → see `RESEARCH.md` for the full synthesis,
competitor intel, and ranked experiment list.** Headline: our diagnosis was right —
all current record-setters use structured construction + local refine. Plan, in
payoff/effort order:
1. Structured basin-hopping in `packer_gpu.py`: vacancy/insertion moves, lattice±defect
   seeds, coherent tilt-cluster moves, ImprovEvolve's 1e3→1e9 penalty ramp (LOW effort)
2. Berthold Farkas-NLP + free SCIP branch-and-bound, warm-started from GPU survivors
   (their recipe took 22 records in May 2026)
3. LLM-evolved move programs (ShinkaEvolve/OpenEvolve) around our JAX evaluator,
   seeded with best-known packings (the ImprovEvolve recipe that beat AlphaEvolve)
4. Dropped after negative evidence: symmetry *constraints* during search (Berthold
   ablation: 539/750 worse), plain CMA-ES/PBO, iid-noise basin-hopping, more restarts.

🏁🏁 2026-07-07 (evening): **BLITZ WAVE 2 COMPLETE — RECORDS #54–56** (pen-in-hex
n=6/8, pen-in-squ n=8, all 6th-decimal beats) from 54 targets / 49 ties / 1 FP /
1 live-gate reject (pen-in-tri n=8: table moved under us — the live-check gate
earned its keep). **THE TESTABLE SPACE IS EXHAUSTED**: all 24 tables harvested
(n≥6), all numerics n=3–8 search-blitzed, plateaus probed, ties documented.
FINAL LEDGER: 56 records — 25 standing, 26 in queue, 7 staged
(READY_blitz_7records, #50–56) + deeper-n=43 note. Vast box: NOTHING LEFT —
destroy. Sustain mode: free L4 on n=21 (frontier saved), daily differ for
competitor counter-moves, repo publication after Erich clears the queue.

🏆×4 2026-07-07 (day): **RECORDS #50–53 — the small-n blitz: pen-in-squ n=3
(2.90811+, the #50 milestone), hex-in-pen n=4/5, oct-in-pen n=4 — all ex-Viquerat,
beaten at the 6th decimal by exhaustive short-run search on n=3–5 entries the sweep's
n≥6 filter had skipped.** Blitz stats: 39 targets, 4 records, 35 ties, <2h GPU. n=21
quad-pincer SHELVED at 1.9987762 (saved_lanes/21_leader.json — 8.6e-5 from Lin, resume
anytime). Retired with verdicts: s24 (near-tie), s38b/s38 (form-locked), h15a/b +
4+4√2 plateau (rigid), L20 (tie), s16 (no-crack). Staged: READY_blitz_4records.
Lesson: WIN thresholds must match Erich's truncation exactly — #50 nearly slipped
through a 1e-5 threshold when the real beat was 2.7e-6.

🏆×11 2026-07-07 (deep night, autonomous): **RECORDS #36–46 — oct-in-tri n=14
(FIRST Viquerat scalp) + n=18; oct-in-hex n=6/20; tri-in-oct n=31–37 (whole tail:
Greenleaf ×2, Metwalli ×5).** Staged folders now: squinhex×2, peninhex×6, peninoct×4,
octintri×2, octinhex×2, triinoct×7 = 23 staged. TALLY: 46 records. Sweep #3 remaining:
hex_in_squ, hex_in_pen, hex_in_oct.

🏆×10 2026-07-06/07 (overnight, autonomous): **RECORDS #26–35 — the pentagon-family
harvest: pen-in-hex n=26–31 (six, full non-Viquerat tail) + pen-in-oct n=28–31 (four).
All ex-Ananthan/Metwalli.** Demographics law confirmed: Ananthan & Metwalli
under-converge (~1e-3 loose); Viquerat & Loyd's non-square pipelines are tight.
Staged: READY_peninhex_n26_to_31, READY_peninoct_n28_to_31 (+ earlier squinhex pair).
TALLY: 35 records (4 standing, 19 pending, 12 staged). Sweep #3 continues: octagon
family → tri_in_oct → hex_in_squ/pen/oct.

🏆×2 2026-07-06: **RECORDS #24–25 — squ-in-hex n=30 (3.6955288, ex-Watson) and n=40
(4.3017139, ex-Loyd)** from sweep #2 on the PNG-named table. tri-in-squ swept DRY
(Loyd/Schadt triangle entries all properly converged — their squares were the loose
pipeline, their triangles weren't). Costantino n=37 GIF = below reconstruction
resolution floor (20px squares, pastel 1px outlines) — parked; GPU lane hunts it.
**HARVEST ERA COMPLETE: every table with usable images swept. 25 records total**
(4 standing, 19 pending, 2 staged). Remaining quests: s45 (Connelly basin → n=44
anomaly), s37 hybrid, 3 tie-verdicts. Email: submission_squinhex_2records_email.txt.

🔬 2026-07-06: **EXACT-SOLVE PIPELINE built (`exact_solve.py`)**: contact-graph
extraction → independent-constraint selection (pivoted QR) → min-norm Gauss-Newton
in mpmath (quadratic convergence 1e-8→1e-80) → PSLQ. First target squ-in-oct n=26:
65-digit value 2.52709499225859076204726543205937697441…, **provably NO closed form**
(no minimal poly ≤ deg 8, coeffs ≤ 1e8; not in Q(√(2+√2)) despite the 22.5° tilt
structure). Structure facts: 4 rattlers, 8 floppy modes (flush-contact sliding),
59 independent contacts. Also acts as a super-polisher (found the basin bottom
3.5e-9 below the f64 value). Squares-only for now (assert nsi==4).

🏁 2026-07-05 (night): **SWEEP COMPLETE. Final campaign tally: 23 RECORDS** —
1 standing (squ-oct 36), 3 sent (squ-tri 41/42/43), 19 staged in 4 batch emails
(tri-pen ×8, squ-oct ×8, tri-hex ×1, hex-hex ×2 incl. #22/#23 n=32/33). Every value
certified ≤1e-12 dilation and verified against live raw HTML. Victims: Lin ×5,
Metwalli ×8, Connelly ×3, Greenleaf ×4, Loyd ×2, Vallejo... The 2026 mass-production
wave was systematically ~1e-3 under-converged; exact-form entries were untouchable.
Still running: Vast lanes (~1.5d credit: s38/L19/s24/L20), GCP s16 (free). Next
session: harvest lane results, send remaining emails, then publish the repo.

🏆×10 2026-07-05 (evening): **RECORDS #12–21 — sweep harvest continued: tri-in-pen
n=42/43 (Greenleaf), tri-in-hex n=36 (Greenleaf), squ-in-oct n=26 (Loyd) + n=31/32/33/
35/38/39/40 (Metwalli's whole June block, margins up to 3.9e-3).** Rejected correctly:
squ-in-oct n=37 (trivial 10−5√2 still holds it) and a stale-CSV n=36 candidate — RULE:
re-base every candidate against LIVE raw HTML, never the scrape snapshot or a WebFetch
summary. Staged emails: triinpen×8, triinhex×1, squinoct×8. TALLY: 21 records
(1 standing, 3 sent, 17 staged). Closed-form entries were untouchable throughout —
the harvest only eats loose numerics.

🏆×6 2026-07-05 (afternoon): **RECORDS #6–11 — the closed-form sweep harvested SIX
tri-in-pen records in one pass: n=28 (2.86636), 29 (2.90241), 30 (2.95014), 34
(3.11073), 35 (3.14843), 36 (3.17473)** — all certified 1e-13, all verified against
raw live HTML (a WebFetch row-misalignment nearly killed three of them; raw HTML is
the arbiter). Email: `submissions/submission_triinpen_6records_email.txt` + 6 renders.
Victims: Vallejo ×2, Loyd, Connelly ×3. The sweep continues on remaining tables.

✅ 2026-07-05: **ERICH ACCEPTED the n=36 improvement — first standing record:
36 squares in an octagon, 2.92870+, credited Aristotle Papowitz.** The n=41/42/43
batch email sent the same day (pending).

⏸️ 2026-07-05 (noon): Session end. RunPod cancelled (results rescued to results/).
Vast (~2 days prepaid, ssh -p 5015 root@103.172.135.141) + GCP L4 (free, packer@<gcp-ip>)
left running: lanes s38/L19/s24/L20/s16 — harvest *_bh_top1.json next session.
TO DO: user sends 3 emails (n=41/42/43 in submissions/); check Erich reply on n=36;
next phase = systematic closed-form sweep (see RECORD #5 note); GCP quota retry for
more free L4s. Total cloud spend this campaign: ~$40. Records: 4 pending + 1 brief.

🏆 2026-07-05 (late morning): **RECORD #5 — 43 squares in a triangle, s = 10.773502699
≈ 5+10/√3 (to 7e-9), beating Lin's 10.77405+ by 5.5e-4.** Same mechanism as #4: his GIF,
our squeeze, an exact form he missed. **THE META-PLAY IS NOW CLEAR: the July-2026
mass-produced entries are frequently loose numerical approximations of exact staircase
forms. Systematic harvest = for every fresh entry, reconstruct GIF → f64 squeeze →
compare against nearby closed forms (a+b/√3 family). Pure CPU, no GPU needed.**
Emails staged for n=41+42+43 (all currently Lin's, all beaten). Insert-into-jammed-form
confirmed dead as a mechanism (n=41→42, 42→43 both failed; vacancy 1.22 < 1.41 needed).

🏆🏆 2026-07-05 (morning): **RECORD #4 — 42 squares in an equilateral triangle,
s = 10.618802157232 ≈ 6+8/√3 (exact form to 4e-9), beating Lin's 10.61956+ by 7.6e-4.**
Found by reconstructing Lin's own GIF and squeezing deeper: his packing was a loose
numeric approximation of an exact construction he didn't recognize. Drop-one then
IMPROVED our own record #3 (n=41 → same closed form, 10.618802157231). Both emails
regenerated (`submissions/submission_n41_email.txt`, `submission_n42_email.txt`).
Cascade continues: n=43 insert attack launched (0.155 headroom to Lin's 10.77405+).
More closed-form IDs from lane telemetry: n=39 record = 8+4/√3 exactly (tie; not
beatable in-basin); n=38 (Costantino 10.304+) is sub-form — his GIF is the seed play.

🏆 2026-07-05 (overnight): **RECORD #3 — 41 squares in an equilateral triangle,
s = 10.618803490699 (10.61880+), beating Haowei Lin's 8-day-old 10.61923+** by 4.3e-4.
Found in <3h of unseeded BH on the "fresh 2026 mass-produced entries are individually
under-converged" thesis. Certified (cost 4e-13), table re-verified, no supersession
(n=42 sits above ours). Email: `submissions/submission_n41_email.txt`. Follow-up n=42
insert attack launched. Negative results banked overnight: all five squ-in-tri
staircase plateau holes are RIGID (hole-rearrangement needs shear-permissive container
walls — octagons yes, triangle corners no); Wu's n=44 basin exhausted (next move:
ask Erich for Connelly's n=45 coords); squ-in-pen n=8/10/14/15 + squ-in-tri n=16/29
reproduced incumbents to 7 digits = documented ties. Lin's squ-in-tri n=20 structure
reconstructed from GIF (non-staircase!) and being chased at 8e-5 above his value.

🏆 2026-07-04 (evening): **RECORD #2 — 36 squares in an octagon, s = 2.928701551520834
(2.92870+), beating the trivial 10−5√2 = 2.92893+.** Found by BH seeded with the exact
trivial-lattice-minus-one (the vacancy rearrangement clicked after ~18 rounds on one
4090). Certified with all margins strictly positive (+5e-11 pair, +1.4e-11 containment,
zero dilation cost). Supersession-checked: best known n=37 (2.92893+) lies ABOVE our
n=36, so no trivial reclaim exists. Email drafted (`submissions/submission_n36_v2_email.txt`).
Solution: `polygon-packer/results/36_record2_polished.json`.

🏅 2026-07-04: **first record set — and lost in 5 minutes.** Our 36-squares-in-octagon
2.929028742 beat Metwalli's 2.92919+ and was accepted by Erich, who then noticed a
trivial 37-diamond lattice at 10−5√2 = 2.92893+ supersedes both n=36 and n=37 (his
words: "you did have the record, for about 5 minutes :)"). Lesson added to the
checklist: **before submitting n, check that no trivial/lattice construction at n+1
beats your value.** We reconstructed his trivial lattice exactly (116 touching pairs,
margins 1e-15; `results/37_trivial_exact.json`) — drop-one squeezes give exactly the
trivial value back (wall-locked lattice), so beating 2.92893 needs rearrangement:
GPU 2/3 BH runs now seeded with lattice-hole variants, targets < 2.92892.

✅ 2026-07-04: tables re-scraped (12 categories, 489 entries → `data/tables-2026-07-04/`),
TARGETS.md rewritten. Headlines: field accelerated (new July-2026 sweepers Haowei Lin,
Bhavithran Ananthan, Derek Wu; records set daily); tri-in-pen n=44 anomaly STILL open
(3.52140+ > n=45's 3.51261+); big legacy blocks intact in squ-in-tri (Friedman 1997,
n=3–36) and hex-in-hex (Morandi 2015); tri-in-hex n=21 improved again (1.99869+, Lin).

The engine, the certification pipeline, the submission channel, and the
competitive map are all in place. What's missing is the constructor — an
algorithms problem, not a compute problem.

## Open item

Upstream PR to Flamethr0wer/polygon-packer with the two bug fixes (SyntaxError
+ infinite loop) — genuinely valuable to that tool's users; needs user
go-ahead (task #7).


## 2026-07-06 (cont.): QUEUE CLEARED — 48 STANDING. ANANTHAN WAVE. BLITZ 3 LAUNCHED.

**ERICH PROCESSED EVERYTHING.** Fresh scrape (data/tables-2026-07-06/, new robust
scraper polygon-packer/scrape_tables.py — handles range rows "41.-42.", picture-less
blocks, closed-form captions): **48 rows now read "Aristotle Papowitz, July 2026"**
(21 earlier + entire 26-batch + squ_in_tri n=42 revealed as range row 41-42).

**ANANTHAN JULY WAVE (24 moves this week).** Bhavithran Ananthan is running the same
under-converged harvest we ran, plus closed forms (squ_in_tri n=39 = 8+4/sqrt3,
n=40 = 7+2sqrt3). He SNIPED 4 of our 7 staged blitz records while they sat unsent:
peninsqu3 (2.90811+), hexinpen4 (3.01521+), octinpen4 (4.10125+), peninhex8
(2.71922+) — all display ties with our certified values, worthless now. LESSON
WRITTEN IN BLOOD: STAGE-AND-SEND SAME DAY. He did NOT touch any of our 48 rows.
Also: Derek Wu improved tri_in_pen n=44 to 3.52140+ (July), narrowing the n=44/45
anomaly. Erich replaced squ_in_oct n=37 with trivial 10-5sqrt2 = 2.92893+.

**3 SURVIVORS staged in submissions/READY_3survivors/ — SEND IMMEDIATELY:**
hexinpen n=5 (3.21972+ vs 3.21973+), peninhex n=6 (2.32293+ vs Ananthan's new
2.322943+), peninsqu n=8 (4.38190+ vs 4.38191+). Live-verified 2026-07-06.
Old 7-record folder archived. Push notification sent to user.

**BLITZ 3 RUNNING on Vast** (69 targets: n=9-12 all categories, ours excluded,
blitz3/ on box, watcher armed): the n<=8 blitzes never covered this band by search
(harvest-only). Early returns: 13 ties + 2 sub-floor sightings vs stale 07-04 claims
that both turned out to be entries Ananthan had ALREADY updated live (oct_in_tri
n=12: ours 13.620323298 vs his 13.62032+ — tie; pen_in_hex n=10: ours 2.936647092
vs his 2.93664+ — tie). Both fully squeezed, no slack. FINAL COMB MUST USE
data/tables-2026-07-06/ FLOORS.

**GCP L4 n=21**: lane had silently restarted fresh (was re-climbing at 1.99887);
killed and RESEEDED from saved_lanes/21_leader.json (1.9987762), target 1.998685,
seed0 4242. NB pkill bracket-trick trap: the LAUNCH text in the same ssh command
line self-matches the pattern — kill and launch in SEPARATE ssh sessions.

**LEDGER after wave: 48 standing + 3 staged survivors + deeper-n=43 note pending.
4 lost to Ananthan. Repo publication: HOLD while Ananthan is active — do not arm him.**


## 2026-07-07 night: HAIL-MARY NIGHT SHIFT (user asleep, ~12h Vast credit left)

User directive: "$20 credits ~ 12h. Formulate a plan to get as many records as
possible. Hail mary, swap targets fast." Blitz 3 final: 69/69, 0 records (n=9-12
fully converged by 2026 crowd incl. AlphaEvolve hex_in_hex n=12).

**NIGHT HARNESS (night/ on Vast box, watcher armed locally):**
- GPU0: night_dropone.py — 398-seed drop-one sweep (every packing JSON we own,
  closest-gap-first, one-level drop-two chaining) vs floors_0707.tsv.
- GPU1: night_recon.py — reconstruct+squeeze the 14 fresh July entries
  (Ananthan's wave, pruned of known-ties); failures/nears feed night/bh_queue.txt;
  then becomes BH worker. NB: box reconstruct_gif.py was stale (no gray-fill
  fallback — July images are gray-filled) and box needed numba — fixed, rerun.
- GPU2: n=21 tri-in-hex BH, batch 1024, 900 rounds, seeded from GCP's 1.9987741.
- GPU3: night_45.sh — THE ANOMALY PLAY: tri_in_pen n=45 (Connelly 3.51261+) is
  8.8e-3 BELOW n=44 (Wu 3.52140+, July). 45_recon.json (on box) -> refine ->
  drop_one -> instant n=44 candidate; then seeded BH; then BH queue worker.
  NB refine_json.py must live in /root/polygon-packer (pack_core import path).
- BH queue workers: 22 rounds/target then rotate (self-rotating).

**READY_3survivors upgraded to 4 items:** added tri_in_pen n=43 self-update
(3.46433+ from 43_deeper_polished.json, certified 3.464339139104347; PNG
rendered page-style). Email + CHECKLIST updated. USER SENDS IN THE MORNING.

**Morning protocol: read night/results.txt BEAT lines -> certify locally
(validate --polish --squeeze) -> live raw-HTML check -> stage READY folder(s) ->
user sends -> DESTROY VAST BOX (credits ~exhausted by then).**


## 2026-07-07 morning: NIGHT SHIFT COMPLETE — 4 RECORDS. VAST BOX DEAD (credits out).

Final: 498 result lines, 4 beats, all certified + staged:
  1. hex_in_pen n=26 = 7.04734+ (7.047340522164712) — recon+squeeze of Ananthan's July entry
  2. pen_in_tri n=26 = 11.45549+ (11.455492110827338) — same, -7.4e-4
  3. tri_in_oct n=34 = 1.86454+ (1.864544399988288) — DROP-ONE from our n=35, -7.2e-3 self-improvement
  4. tri_in_oct n=33 = 1.84789+ (1.847892538040437) — cascade from new 34, -4.4e-3
KEY MECHANISM PROVEN: better-basin propagation — drop-one from a strong n+1
basin beats weaker-architecture records below (our 33/34 were Metwalli-derived;
our 35 was not). Cascade halts when the lower record is already strong (32).

Negative results (definitive): drop-one sweep over ~400 seeds — only the
tri_in_oct cascade hit; all other neighbors tight (validated: reproduced 17/9
honeycomb + closed forms to 1e-7). n=44/45 anomaly NOT recoverable from image
(3 attempts; recon lands ~3.53 vs Connelly's 3.5126) → P.S. in morning email
asks Erich for Connelly's coordinates. Final BH ties of note: oct_in_oct 24 at
+8.4e-6, oct_in_hex 18 at +4.5e-6 above Ananthan floors (his values converged).
n=21: no progress past 1.9987727 (420+ total rounds) — needs a new idea.

STAGED FOR SEND: submissions/READY_MORNING_batch6/ (6 items incl. night finds
1-2 + n43 self-update + anomaly P.S.) + READY_night_trioct_cascade/ (records
3-4). Harvest of everything: polygon-packer/results/night_harvest/ (1200+ JSONs).
GCP L4 (free) still running n=21. Vast box GONE — nothing to destroy.
