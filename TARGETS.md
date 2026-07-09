# Packing Records — Target List (re-scraped 2026-07-04)

*Full re-scrape of 12 polygon-in-polygon category pages (489 entries), verified against
raw HTML. Most pages carry a site-index update date of **7/3/26** — the table moves
every few days. Raw data: `data/tables-2026-07-04/*.csv` + `analysis.txt` (regenerate
with `analyze_tables.py`). Previous version of this file was compiled 2026-07-03 and
was already stale.*

## How records work (unchanged, re-verified)

- **Submission = email Erich Friedman: `erichfriedman68@gmail.com`.** Channel proven
  (our n=8 submission got a same-day reply). Include: category, n, claimed s (6+ digits,
  truncated), full-precision coordinates JSON, image, brief method note, real name.
- **Format**: `s = 2.32251+` (truncated) or exact closed forms. Site convention:
  container side in units of inner-polygon side; in packer terms
  `s = S·sin(π/nsc)/sin(π/nsi)`, S = container circumradius.
- Category grid: 24 hosted pages (5 shapes × 5 containers, minus squares-in-squares
  which is external to David Ellsworth's page). Slugs: `triinsqu`, `squinpen`,
  `hexinhex`, etc. at `https://erich-friedman.github.io/packing/<slug>/`.

## The competition (holder counts across our 12 scraped categories)

| Who | Entries | Notes |
|---|---|---|
| Jake Loyd | 97 | June 2026 mass-producer; owns large-n tails everywhere |
| Erich Friedman | 54 | mostly legacy 1996–2015 closed forms |
| Maurizio Morandi | 47 | legacy 2008–2015 blocks + still active (May/Jun 2026) |
| Timo Berthold et al | 26 | the arXiv:2605.04850 NLP-solver records, accepted Jan–May 2026 |
| **Haowei Lin** | 24 | **NEW, July 2026 — actively sweeping right now** |
| Jonathan Viquerat | 19 | May–Jun 2026 |
| Ignacio Vallejo | 19 | polygon-packer author, Mar 2026 |
| David W. Cantrell | 19 | legacy 2002–2012 |
| **Bhavithran Ananthan** | 18 | **NEW, July 2026 — actively sweeping** |
| Emerson Connelly | 18 | Apr–May 2026 |
| Ian Watson | 14 | Apr 2026 |
| Schadt 12 · Greenleaf 12 · Metwalli 10 · Kravatsky 7 · Mowry 4 · Kamp 3 | | + Derek Wu, Frenzley, Vesterinen, Costantino, Leach (Jul 2026) |

**Read**: the June wave never stopped — it accelerated. Records set *this week*
(Lin, Ananthan, Wu: July 2026 dates everywhere). Anything fresh will be re-attacked
within days; legacy closed-form entries and structural anomalies are the durable targets.
"Papowitz" appears nowhere yet — our n=8 tie was correctly not listed.

## Tier 0 — anomalies (free slack proven by monotonicity)

- **Triangles in pentagon, n=44 — STILL STANDS**: n=44 = 3.52140+ (Derek Wu, Jul 2026)
  is WORSE than n=45 = 3.51261+ (Emerson Connelly, May 2026). The only strict
  monotonicity violation across all 489 scraped entries. A ≤3.51261 packing of 44
  exists (Connelly's 45 minus one triangle) — but no coordinates are published, so we
  must produce it: reconstruct n=45 from its GIF + polish, or run structured search
  at n=44. Note Wu already shaved this in July (was 3.52387 Greenleaf) and missed the
  obvious fix — window is open but others are circling.
- **Squares in octagon, n=36 near-plateau**: n=36 = 2.92919+ and n=37 = 2.92926+
  (both Metwalli, Jun 2026) differ by 7×10⁻⁵. A 37th square nearly free ⇒ n=36 very
  likely has real slack. Attack n=36 (and n=37 falls with it).

## Tier 1 — legacy blocks still standing (2015 or older, moderate n)

⚠️ Our own campaign lesson tempers this tier: tri-in-hex legacy entries n=8,16,17,18
proved to be *fully converged* basins (true ties). Legacy ≠ soft. But the 2026 wave
keeps displacing 1996–2012 entries (tri-in-squ n=12 fell to Kamp in Apr, squ-in-tri
n=20/22 fell in May/Jul), so parts of these blocks ARE soft — the play is cheap GPU
waves per entry, expect ties, harvest outliers.

- **Squares in triangle** — the biggest intact legacy block: Friedman-1997 holds
  n=3–36 nearly wholesale (Cantrell 2002 at n=8,12,17,23,30). The 2026 shaves at
  n=20, 22, 37–43 prove the staircase pattern is beatable. Prime sub-targets: the
  closed-form staircase entries n=16 (6+2/√3 = 7.155) and n=29 (8+2/√3 = 9.155) —
  analogous entries n=22, n=37 both fell in 2026. Also plateau pairs 26–27 (8.618+),
  33–34 (9.618+), 35–36 (9.773+): lower member of each pair likely has slack.
- **Triangles in square**: still-standing legacy at n=5,6,7,8 (Friedman 1996!),
  n=9,11,17,18,26 (Morandi 2008), n=10,13,14 (Cantrell 2002), n=25 (Cantrell 2012).
- **Triangles in triangle**: legacy at n=5,10,11,18,27,28 (Friedman 1997), n=12,29
  (Cantrell 2007), n=6,13,17,19–22,30 (Morandi 2008).
- **Hexagons in hexagon**: Morandi-2015 block n=6,7,8,9,10,18,19,22,24 + Friedman
  n=5 (8/3). Note the *non-trivial plateaus*: n=6–7 both 5/√3 = 2.886+ and n=18–19
  both 8/√3 = 4.618+ — the lower member (6, 18) of each pair plausibly has slack.
  AlphaEvolve still holds n=12 (3.94164+, May 2025) — beating DeepMind remains a story.
  n=11 is Schellhorn Jul 2025 (2+10/3√3 = 3.92450+, closed form — likely rigid).
- **Squares in pentagon**: 2012-era survivors n=5,6,7 (Friedman), n=8,14,15 (Cantrell),
  n=4 (Morandi). n=9–13 fell to Berthold/Loyd in 2026 — the neighbors moving proves
  the region was soft.
- **Triangles in pentagon**: 2012 survivors n=3,4 (Morandi), n=5 (Friedman),
  n=7 (Cantrell), n=8,9 (Morandi/Cantrell).
- **Triangles in hexagon**: legacy n=7 (1.277+ Cantrell 2005), n=9 (1.434+ Friedman
  2005), n=14 (1.725+ Cantrell 2012). We already proved n=8,16,17,18 are ties; n=11
  (3/2) and n=15 (7/4) are exact fractions, likely rigid. n=10,12 fell to Berthold,
  n=13 to Watson (2026) — the era of soft entries here is mostly over.
- **Hexagons in triangle**: n=5 (Morandi 2015), n=8 (25/3, Friedman 2015).
- **Squares in octagon**: 2013 survivors n=3 (Friedman), n=6,7,12 (Morandi).

## Tier 2 — trivial plateaus (KNOWN HARD — structured moves only)

Confirmed by our n=21/22 tri-in-hex experiments: winning configs live in tiny
globally-rearranged basins that random multi-start never finds (n=21's record was
found — and improved to 1.99869+ by Haowei Lin in July 2026 — by others, never by
our 4,700+ restarts). Attack only with the RESEARCH.md structured-move arsenal.

- tri-in-hex: n=22–24 all s=2 (n=21 = 1.99869+ proves sub-2 exists nearby);
  n=50–54 all s=3.
- tri-in-tri: k²−1/k²−2 entries — n=7,8 (s=3), 14,15 (s=4), 23,24 (s=5), 34,35 (s=6),
  47,48 (s=7).
- squ-in-oct: n=16–17 both 5√2−5 = 2.07106+ (17 fits in 16's container!); n=24
  (6√2−6 = 2.48528+, trivial).
- hex-in-hex: n=13 (s=4), 20–21 (s=5), 29–31 (s=6).
- hex-in-tri: n=14–15 (6√3), 20–21 (7√3), 26–28 (see typo note below).

## Avoid — fresh 2026 blocks under active sweep

pen-in-tri (39/40 entries are 2026), oct-in-oct (26/30), squ-in-hex (34/40, page is
weeks old), all large-n tails held by Loyd/Lin/Ananthan. These move within days;
anything we'd find would likely be re-beaten before Erich processes the email.
Exception: oct-in-oct n=29 (Vallejo 6.05746+, Mar 2026) still stands — beating the
tool author with his own tool remains a nice flag, low priority.

## Site typo worth reporting (goodwill + name recognition)

hex-in-tri n=26–28: page HTML says "s = 7√3 = 13.85677+" — but 7√3 ≈ 12.124 while
the decimal matches 8√3 ≈ 13.856. Emailing Erich the correction (politely, separate
from any record claim) builds the relationship. Raw HTML snapshot preserved in the
scrape agent's files if needed.

## Pipeline status

1. ✅ Engine: `packer_gpu.py` (batched GPU search + f64 refine), `validate_packing.py`
   (independent certification) — see CAMPAIGN.md.
2. ✅ Submission channel proven (n=8 tie submitted, Erich replied).
3. ⏳ NEXT (per RESEARCH.md): structured basin-hopping moves (vacancy/insertion,
   lattice±defect seeds, coherent tilt-cluster kicks, 1e3→1e9 penalty ramp), then
   SCIP/Farkas warm-started global solves, then LLM-evolved move programs.

## Submission checklist (when we beat something)

1. `validate_packing.py --polish --squeeze` → certified size beats table value.
2. Render clean PNG + full-precision JSON coordinates.
3. **Re-verify the live table entry the same day** — it may have moved (records are
   currently being set every few days).
4. Email erichfriedman68@gmail.com: category, n, claimed s (6+ digits, truncated),
   coordinates, image, brief method note. Credit: Aristotle Papowitz.
