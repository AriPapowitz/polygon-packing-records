# Attack matrix — what the campaign examined, and with what outcome

Reconstructed 2026-08-17 from local evidence (sweep logs, blitz target and
result files, night-harvest artifacts, campaign log, frozen ledger).
Machine-readable: `paper/attack_matrix.csv`. Regenerate:
`python paper/build_attack_matrix.py`.

## Denominators

- **551 distinct (category, n) problems examined** by at least one mechanism.
- **54** beaten and claimed (the ledger's 54).
- **5** beaten but never claimed (sniped, rejected, or superseded before submission).
- **385** documented ties (our refined value logged, no improvement possible in basin).
- **49** searched with no improvement (per-entry or aggregate-logged).
- **3** attempt failed (reconstruction/refine failure or live-gate reject).
- **55** attempted, outcome not logged.

## Examined vs beaten, by incumbent at time of attempt

| Incumbent | Examined | Beaten | Documented ties |
|---|---|---|---|
| Jake Loyd | 145 | 3 | 139 |
| Jonathan Viquerat | 117 | 7 | 94 |
| Bhavithran Ananthan | 42 | 14 | 24 |
| Timo Berthold et al | 33 | 0 | 26 |
| Maurizio Morandi | 30 | 0 | 10 |
| Mohamed Metwalli | 22 | 19 | 2 |
| Trivial | 18 | 0 | 0 |
| Erich Friedman | 17 | 0 | 2 |
| Haowei Lin | 16 | 3 | 13 |
| Ignacio Vallejo | 16 | 2 | 12 |
| Ian Watson | 13 | 1 | 12 |
| Nicolas Campailla | 13 | 1 | 10 |
| David W. Cantrell | 11 | 0 | 4 |
| Thomas Greenleaf | 11 | 6 | 5 |
| Grant Mowry | 8 | 0 | 7 |
| Alex Kravatsky et al | 7 | 0 | 7 |
| Emerson Connelly | 7 | 3 | 0 |
| Thomas Schadt | 6 | 0 | 6 |
| Dominik Kamp | 3 | 0 | 2 |
| Maksymilian Jankowski | 3 | 0 | 3 |

Caveats: targeting was not uniform (fresh, suspected-loose entries were
prioritized; some blocks were deliberately avoided as actively swept), so
these are audit coverage numbers, not unbiased sampling rates. Rows marked
'attempted, outcome not logged' had compute spent but no per-entry outcome
recorded; aggregate-only outcomes are marked in the evidence column.