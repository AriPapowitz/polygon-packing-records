# Record Ledger — frozen 2026-08-16

Every problem claimed in the July 2026 campaign, one row per (category, n) cell,
reconciled against dated scrapes of the live tables (2026-07-04, 2026-07-06, 2026-07-07, 2026-07-07b, 2026-07-09, 2026-08-16)
and re-certified from the submitted coordinates on 2026-08-16.
Regenerate: `python paper/collect_and_certify.py` then `python paper/build_ledger.py`
(machine-readable: `paper/ledger.csv`, `paper/certification.csv`; coordinates:
`paper/solutions/`).

## Headline numbers

Conservative — only what the archives and a fresh certification run can prove.

- **54 problems claimed** across 14 tables, submitted July 4–8, 2026 in
  58 claim line-items (4 cells were submitted twice; see notes).
- **54/54 re-certified on 2026-08-16** from the exact coordinates that were
  emailed: independent exact-SAT certification reproduces every claimed value to
  ≤1e-9; worst pair separation ≥ −8e-13, worst containment ≥ −6e-13,
  certification cost ≤ 1e-11 (`paper/certification.csv`).
- **52 problems credited** "Aristotle Papowitz, July 2026" — *verified* in at
  least one archived scrape.
- **41 standing as of 2026-08-16** (49 were standing 2026-07-09).
- **11 since re-taken**: Bhavithran Ananthan ×5, Aapo Lipponen ×4, Leo Strijbos ×1, Luke Kaiser ×1.
- **2 claims excluded from the credited count**:
  - pen_in_oct 28 — the campaign log records the full 26-batch being processed,
    but a competitor had posted a better value the day the batch went out and no
    archived scrape captures our listing. Conservatively: not counted.
  - pen_in_hex 6 — withdrawn: the page we verified against was stale; the
    incumbent's actual value was not beaten. Not a record.
- Not counted anywhere: tri_in_hex 8 (July 2–3 channel test) — certified
  1.356597399687052, a **tie** with Cantrell 2012, correctly never listed.

**Reconciling the campaign's live count of "58 records on the page":** during the
campaign the tables showed, at one point or another, the 52 verified credits plus
squ_in_oct 36's first listing (~5 minutes, then superseded by the trivial lattice
and re-won), pen_in_oct 28's brief listing, and re-listings of tri_in_pen 43 and
tri_in_oct 33/34 at improved values. The paper claims the conservative,
scrape-verifiable number: **52**.

Mechanisms (per problem, final value): harvest ×45, search ×4, closed-form ×2, drop-one ×2, search+drop-one ×1.

| # | Mechanism | Meaning |
|---|---|---|
| 1 | harvest | reconstruct incumbent's published image (~0.1 px), converge it in float64 below their displayed floor |
| 2 | search | new arrangement found by batched GPU multi-start / structured basin-hopping |
| 3 | closed-form | harvest that landed on an exact algebraic value the incumbent missed (confirmed by 160-digit exact-solve + PSLQ) |
| 4 | drop-one | remove one shape from a stronger (n+1)-packing, re-squeeze; cascades downward |

## The ledger

Margin = strongest prior displayed floor − our certified value (a lower bound on
the improvement, since a displayed `x+` means a value in [x, x+10⁻⁵)). ✓ = the
submitted coordinates re-certified to the claimed value on 2026-08-16.

| Category | n | Ours (certified) | Prior entry | Margin ≥ | Mech | Credited (scrape) | ✓ | Status (2026-08-16) |
|---|---|---|---|---|---|---|---|---|
| Hexagons in Hexagons | 32 | 6.26088195827499 | 6.26164+ (Bhavithran Ananthan) | 7.6e-04 | harvest | 2026-07-06 | ✓ | standing |
| Hexagons in Hexagons | 33 | 6.26589678842626 | 6.26603+ (Thomas Greenleaf) | 1.3e-04 | harvest | 2026-07-06 | ✓ | standing |
| Hexagons in Pentagons | 5 | 3.21972844123009 | 3.21973+ (Jonathan Viquerat) | 1.6e-06 | search | 2026-07-09 | ✓ | standing |
| Hexagons in Pentagons | 26 | 7.04734052216266 | 7.04739+ (Bhavithran Ananthan) | 4.9e-05 | harvest | 2026-07-09 | ✓ | standing |
| Hexagons in Pentagons | 27 | 7.15307144949031 | 7.15354+ (Bhavithran Ananthan) | 4.7e-04 | harvest | 2026-07-06 | ✓ | standing |
| Hexagons in Pentagons | 30 | 7.45985359687768 | 7.46147+ (Mohamed Metwalli) | 1.6e-03 | harvest | 2026-07-06 | ✓ | standing |
| Hexagons in Pentagons | 32 | 7.76649522200506 | 7.76651+ (Bhavithran Ananthan) | 1.5e-05 | harvest | 2026-07-06 | ✓ | re-taken by Bhavithran Ananthan (July 2026) |
| Octagons in Hexagons | 6 | 3.94730630238847 | 3.9496+ (Mohamed Metwalli) | 2.3e-03 | harvest | 2026-07-06 | ✓ | standing |
| Octagons in Hexagons | 20 | 6.9945048050501 | 6.99466+ (Bhavithran Ananthan) | 1.6e-04 | harvest | 2026-07-06 | ✓ | re-taken by Leo Strijbos (August 2026) |
| Octagons in Triangles | 14 | 14.2392141111 | 14.24006+ (Jonathan Viquerat) | 8.5e-04 | harvest | 2026-07-06 | ✓ | standing |
| Octagons in Triangles | 18 | 16.409792942504 | 16.41137+ (Bhavithran Ananthan) | 1.6e-03 | harvest | 2026-07-06 | ✓ | standing |
| Pentagons in Hexagons | 6 | 2.32293929566567 | 2.322943+ (Bhavithran Ananthan) | 3.7e-06 | search | — | ✓ | withdrawn — not a record (page was stale; the incumbent's actual value was not beaten) |
| Pentagons in Hexagons | 26 | 4.66031300394371 | 4.66044+ (Bhavithran Ananthan) | 1.3e-04 | harvest | 2026-07-06 | ✓ | standing |
| Pentagons in Hexagons | 27 | 4.71756610082141 | 4.718+ (Bhavithran Ananthan) | 4.3e-04 | harvest | 2026-07-06 | ✓ | standing |
| Pentagons in Hexagons | 28 | 4.8439971285698 | 4.8441+ (Bhavithran Ananthan) | 1.0e-04 | harvest | 2026-07-06 | ✓ | re-taken by Bhavithran Ananthan (July 2026) |
| Pentagons in Hexagons | 29 | 4.91003716809624 | 4.91275+ (Mohamed Metwalli) | 2.7e-03 | harvest | 2026-07-06 | ✓ | standing |
| Pentagons in Hexagons | 30 | 4.96406550325769 | 4.96723+ (Mohamed Metwalli) | 3.2e-03 | harvest | 2026-07-06 | ✓ | standing |
| Pentagons in Hexagons | 31 | 5.09318706622837 | 5.09328+ (Bhavithran Ananthan) | 9.3e-05 | harvest | 2026-07-06 | ✓ | re-taken by Bhavithran Ananthan (July 2026) |
| Pentagons in Octagons | 28 | 3.56575100826471 | 3.56685+ (Mohamed Metwalli) | 1.1e-03 | harvest | — | ✓ | briefly listed during 26-batch processing, re-taken the same day — not captured in any archived scrape |
| Pentagons in Octagons | 29 | 3.61896400547029 | 3.61924+ (Bhavithran Ananthan) | 2.8e-04 | harvest | 2026-07-06 | ✓ | re-taken by Bhavithran Ananthan (July 2026) |
| Pentagons in Octagons | 30 | 3.6667804456925 | 3.66702+ (Bhavithran Ananthan) | 2.4e-04 | harvest | 2026-07-06 | ✓ | standing |
| Pentagons in Octagons | 31 | 3.74416279501696 | 3.74457+ (Bhavithran Ananthan) | 4.1e-04 | harvest | 2026-07-06 | ✓ | re-taken by Bhavithran Ananthan (July 2026) |
| Pentagons in Squares | 8 | 4.38190960645924 | 4.38191+ (Jonathan Viquerat) | 3.9e-07 | search | 2026-07-09 | ✓ | standing |
| Pentagons in Triangles | 26 | 11.455492110824 | 11.45623+ (Bhavithran Ananthan) | 7.4e-04 | harvest | 2026-07-09 | ✓ | standing |
| Squares in Hexagons | 30 | 3.69552878973383 | 3.69554+ (Ian Watson) | 1.1e-05 | harvest | 2026-07-06 | ✓ | standing |
| Squares in Hexagons | 40 | 4.30171393148465 | 4.30173+ (Jake Loyd) | 1.6e-05 | harvest | 2026-07-06 | ✓ | standing |
| Squares in Octagons | 26 | 2.52709499225859 | 2.52711+ (Jake Loyd) | 1.5e-05 | harvest | 2026-07-06 | ✓ | standing |
| Squares in Octagons | 31 | 2.7842807351438 | 2.7857+ (Mohamed Metwalli) | 1.4e-03 | harvest | 2026-07-06 | ✓ | re-taken by Aapo Lipponen (July 2026) |
| Squares in Octagons | 32 | 2.83980869233204 | 2.84222+ (Mohamed Metwalli) | 2.4e-03 | harvest | 2026-07-06 | ✓ | standing |
| Squares in Octagons | 33 | 2.85903637743338 | 2.8629+ (Mohamed Metwalli) | 3.9e-03 | harvest | 2026-07-06 | ✓ | standing |
| Squares in Octagons | 35 | 2.90679519404119 | 2.91052+ (Mohamed Metwalli) | 3.7e-03 | harvest | 2026-07-06 | ✓ | re-taken by Aapo Lipponen (July 2026) |
| Squares in Octagons | 36 | 2.92870155152083 | 2.92893+ (Trivial (10−5√2)) | 2.3e-04 | search | 2026-07-06 | ✓ | standing |
| Squares in Octagons | 38 | 3.02113017873058 | 3.02321+ (Mohamed Metwalli) | 2.1e-03 | harvest | 2026-07-06 | ✓ | re-taken by Luke Kaiser (August 2026) |
| Squares in Octagons | 39 | 3.08444308203313 | 3.08686+ (Mohamed Metwalli) | 2.4e-03 | harvest | 2026-07-06 | ✓ | re-taken by Aapo Lipponen (July 2026) |
| Squares in Octagons | 40 | 3.14201757666269 | 3.14545+ (Mohamed Metwalli) | 3.4e-03 | harvest | 2026-07-06 | ✓ | re-taken by Aapo Lipponen (July 2026) |
| Squares in Triangles | 41 | 10.6188021572304 | 10.61923+ (Haowei Lin) | 4.3e-04 | search+drop-one | 2026-07-06 | ✓ | standing |
| Squares in Triangles | 42 | 10.6188021572304 | 10.61956+ (Haowei Lin) | 7.6e-04 | closed-form | 2026-07-06 | ✓ | standing |
| Squares in Triangles | 43 | 10.773502698923 | 10.77405+ (Haowei Lin) | 5.5e-04 | closed-form | 2026-07-06 | ✓ | standing |
| Triangles in Hexagons | 36 | 2.58307812035605 | 2.58309+ (Thomas Greenleaf) | 1.2e-05 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Octagons | 31 | 1.80338022858235 | 1.80341+ (Thomas Greenleaf) | 3.0e-05 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Octagons | 32 | 1.82994256115661 | 1.83002+ (Thomas Greenleaf) | 7.7e-05 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Octagons | 33 | 1.84789253804022 | 1.85532+ (Mohamed Metwalli) | 7.4e-03 | drop-one | 2026-07-06 | ✓ | standing |
| Triangles in Octagons | 34 | 1.86454439998733 | 1.87179+ (Mohamed Metwalli) | 7.2e-03 | drop-one | 2026-07-06 | ✓ | standing |
| Triangles in Octagons | 35 | 1.89051358097131 | 1.89225+ (Mohamed Metwalli) | 1.7e-03 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Octagons | 36 | 1.92667874651466 | 1.92843+ (Mohamed Metwalli) | 1.8e-03 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Octagons | 37 | 1.95615978426801 | 1.95803+ (Mohamed Metwalli) | 1.9e-03 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Pentagons | 28 | 2.86636440571096 | 2.868+ (Ignacio Vallejo) | 1.6e-03 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Pentagons | 29 | 2.90240708611043 | 2.903+ (Ignacio Vallejo) | 5.9e-04 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Pentagons | 30 | 2.95014178509158 | 2.9502+ (Jake Loyd) | 5.8e-05 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Pentagons | 34 | 3.11073148512174 | 3.114+ (Emerson Connelly) | 3.3e-03 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Pentagons | 35 | 3.14843301598468 | 3.15+ (Emerson Connelly) | 1.6e-03 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Pentagons | 36 | 3.17473090525275 | 3.177+ (Emerson Connelly) | 2.3e-03 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Pentagons | 42 | 3.42505733584122 | 3.42525+ (Thomas Greenleaf) | 1.9e-04 | harvest | 2026-07-06 | ✓ | standing |
| Triangles in Pentagons | 43 | 3.46433913910293 | 3.46441+ (Thomas Greenleaf) | 7.1e-05 | harvest | 2026-07-06 | ✓ | standing |

## Attrition timeline

| Scrape | Standing | Event |
|---|---|---|
| 2026-07-04 | 0 | campaign start: tables re-scraped, no credits yet |
| 2026-07-06 | 48 | queue cleared: batches B1–B6 processed |
| 2026-07-07 | 48 | blitz-era snapshot |
| 2026-07-07b | 48 | night-shift snapshot |
| 2026-07-09 | 49 | B7 processed (7 of 8 listed); Lipponen takes squ_in_oct 31/35/40 |
| 2026-08-16 | 41 | ledger freeze; attrition: Bhavithran Ananthan (5), Aapo Lipponen (4), Leo Strijbos (1), Luke Kaiser (1) |

## Notes for the paper

1. **Truncation semantics.** All margins are lower bounds against the displayed
   floor. Every claim was certified before submission and re-certified for this
   ledger (exact separating-axis margins, float64, certified dilation bound).
2. **The five-minute record.** squ_in_oct 36's first value (2.92902...) beat the
   then-listed 2.92919+ but was superseded within minutes by a trivial 37-square
   diamond lattice at 10−5√2 = 2.92893+ dominating both n=36 and n=37; the
   standing value (2.92870...) then beat the trivial lattice with a vacancy
   rearrangement. Lesson: check trivial n+1 constructions before claiming n.
3. **Race dynamics.** Two claims were invalidated purely by table velocity
   (pen_in_hex 6, pen_in_oct 28); four further finished records (pen_in_squ 3,
   hex_in_pen 4, oct_in_pen 4, pen_in_hex 8) were sniped by a competitor while
   staged, before sending, and were never submitted. Records on a live benchmark
   decay: 52 credited → 49 standing Jul 9 → 41 standing Aug 16.
4. **Resubmitted cells.** squ_in_oct 36 (after the trivial-lattice supersession),
   tri_in_pen 43 (display-identical deeper certification), tri_in_oct 34 and 33
   (drop-one self-improvements of our own week-old values, −7.2e-3 and −4.4e-3).
5. **Exact forms (confirmed 2026-08-16).** The contact systems of squ_in_tri
   41/42 and 43 solve exactly to **6+8/√3** and **5+10/√3** (160-digit
   contact-manifold Gauss–Newton, quadratic convergence to |F| < 1e-160; PSLQ
   minimal polynomials 3s²−36s+44 and 3s²−30s−25; see
   `polygon-packer/results/exact_squ_in_tri_4*.md`). The submitted f64 values
   sit 3.7e-9 / 7.0e-9 above the exact bottoms — identical truncated display.
   Structure: n=43 has 25/43 load-bearing squares (13-dim floppy manifold);
   n=42/41 have 39/38 load-bearing (9-dim).
6. **Provenance.** Claimed values and coordinates: batch emails + coordinate
   files in `submissions/sent_batches/` (private), frozen as JSON in
   `paper/solutions/` (public). Table history: `data/tables-*/`. Certification:
   `paper/certification.csv`, regenerable with `paper/collect_and_certify.py`.
