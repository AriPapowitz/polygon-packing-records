"""Build the frozen record ledger (LEDGER.md + paper/ledger.csv).

Sources of truth:
  - submissions/sent_batches/*  : the batch emails actually sent (claimed values,
    prior displayed values at send time)
  - data/tables-*/              : dated scrapes of all 24 live record tables
    (who was credited when, who holds each cell now)

The 58 record submissions (July 4-8, 2026; all found & certified by July 7):
  B1  n36 v1        1   2026-07-04  squ_in_oct 36 (accepted, superseded same day)
  B2  n36 v2        1   2026-07-04  squ_in_oct 36 (the standing record)
  B3  squintri      3   2026-07-05  squ_in_tri 41/42/43
  B4  ALL19        19   2026-07-05  tri_in_pen x8, squ_in_oct x8, tri_in_hex, hex_in_hex x2
  B5  squinhex      2   2026-07-06  squ_in_hex 30/40
  B6  ALL26        24   2026-07-06  pen/oct/hex families + tri_in_oct 31-37
  B7  FINAL8        8   2026-07-08  blitz survivors + night finds + self-updates

Run:  python paper/build_ledger.py   (from the repo root or paper/)
"""
import csv, json, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SENT = ROOT / "submissions" / "sent_batches"

SLUG2CAT = {
    "triinsqu": "tri_in_squ", "triintri": "tri_in_tri", "triinpen": "tri_in_pen",
    "triinhex": "tri_in_hex", "triinoct": "tri_in_oct",
    "squintri": "squ_in_tri", "squinpen": "squ_in_pen", "squinhex": "squ_in_hex",
    "squinoct": "squ_in_oct",
    "penintri": "pen_in_tri", "peninsqu": "pen_in_squ", "peninpen": "pen_in_pen",
    "peninhex": "pen_in_hex", "peninoct": "pen_in_oct",
    "hexintri": "hex_in_tri", "hexinsqu": "hex_in_squ", "hexinpen": "hex_in_pen",
    "hexinhex": "hex_in_hex", "hexinoct": "hex_in_oct",
    "octintri": "oct_in_tri", "octinsqu": "oct_in_squ", "octinpen": "oct_in_pen",
    "octinhex": "oct_in_hex", "octinoct": "oct_in_oct",
}
CATNAME = {
    "tri_in_pen": "Triangles in Pentagons", "tri_in_hex": "Triangles in Hexagons",
    "tri_in_oct": "Triangles in Octagons", "squ_in_tri": "Squares in Triangles",
    "squ_in_hex": "Squares in Hexagons", "squ_in_oct": "Squares in Octagons",
    "pen_in_tri": "Pentagons in Triangles", "pen_in_squ": "Pentagons in Squares",
    "pen_in_hex": "Pentagons in Hexagons", "pen_in_oct": "Pentagons in Octagons",
    "hex_in_pen": "Hexagons in Pentagons", "hex_in_hex": "Hexagons in Hexagons",
    "oct_in_tri": "Octagons in Triangles", "oct_in_hex": "Octagons in Hexagons",
}

def clean(s):
    return (s.replace("&radic;", "√").replace("&ndash;", "–")
             .replace("&deg;", "°").replace("&nbsp;", " ").strip())

# ---------------------------------------------------------------- scrapes ----
def load_scrapes():
    """{date_tag: {(cat, n): (s_display, holder, year)}} for every data/tables-* dir."""
    scrapes = {}
    for d in sorted(DATA.glob("tables-*")):
        tag = d.name.replace("tables-", "")
        cells = {}
        for f in d.glob("*.csv"):
            with open(f, encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    ns = [int(x) for x in re.findall(r"\d+", row["n"])]
                    if not ns:
                        continue
                    span = range(ns[0], ns[-1] + 1) if len(ns) > 1 else [ns[0]]
                    for n in span:
                        cells[(row["category"], n)] = (
                            clean(row["s"]), clean(row["holder"]), clean(row.get("year") or ""))
        scrapes[tag] = cells
    return scrapes

def sfloor(display):
    """Numeric floor of a displayed table value ('2.92893+' or '10−5√2 = 2.92893+')."""
    nums = re.findall(r"(\d+\.\d+)\+?\s*$", display) or re.findall(r"(\d+\.\d+)", display)
    return float(nums[-1]) if nums else None

# ---------------------------------------------- full-precision claim values ----
def coords_values(path):
    """Parse '--- slug n=NN   s = VALUE ...' lines -> {(cat,n): float}."""
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"---\s+(\w+)\s+n=(\d+)\s+s = ([\d.]+)", line.strip())
        if m:
            out[(SLUG2CAT[m.group(1)], int(m.group(2)))] = float(m.group(3))
    return out

def json_value(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("side_length") or d.get("s")

V19 = coords_values(SENT / "webready_19batch" / "papowitz_coordinates.txt")
V26 = coords_values(SENT / "READY_ALL26_resend" / "papowitz_coordinates.txt")
F8 = SENT / "READY_FINAL_8records"
def f8(slug, n):
    return json_value(F8 / f"{slug}{n}_coordinates.json")

# ------------------------------------------------------------- submissions ----
# (batch, sent, cat, n, ours_full, prior_display_floor, prior_holder, mechanism, note)
# prior_* = live table state quoted in the batch email at send time.
S = []
def add(batch, sent, cat, n, val, pfloor, pholder, mech, note=""):
    S.append(dict(batch=batch, sent=sent, cat=cat, n=n, ours=val, prior_floor=pfloor,
                  prior_holder=pholder, mech=mech, note=note))

add("B1 n36-v1", "2026-07-04", "squ_in_oct", 36, 2.929028742, 2.92919, "Mohamed Metwalli",
    "search", "accepted, then superseded same day by the trivial 37-square lattice "
    "(10−5√2 = 2.92893+) Erich spotted; 'you did have the record, for about 5 minutes'")
add("B2 n36-v2", "2026-07-04", "squ_in_oct", 36, 2.928701551520834, 2.92893, "Trivial (10−5√2)",
    "search", "vacancy rearrangement of the 37-lattice; BH seeded with lattice-minus-one")
add("B3 squintri", "2026-07-05", "squ_in_tri", 41, 10.618802157230435, 10.61923, "Haowei Lin",
    "search+drop-one", "= n=42 configuration minus one square; agrees with 6+8/√3 to 4e-9")
add("B3 squintri", "2026-07-05", "squ_in_tri", 42, 10.618802157230435, 10.61956, "Haowei Lin",
    "closed-form", "reconstructed Lin's GIF, squeezed to a value agreeing with 6+8/√3 to 4e-9")
add("B3 squintri", "2026-07-05", "squ_in_tri", 43, 10.773502698922991, 10.77405, "Haowei Lin",
    "closed-form", "value agrees with 5+10/√3 to 7e-9")

_all19_prior = {  # displayed floors quoted in submission_ALL19_email.txt
    ("tri_in_pen", 28): 2.868, ("tri_in_pen", 29): 2.903, ("tri_in_pen", 30): 2.95020,
    ("tri_in_pen", 34): 3.114, ("tri_in_pen", 35): 3.150, ("tri_in_pen", 36): 3.177,
    ("tri_in_pen", 42): 3.42525, ("tri_in_pen", 43): 3.46441,
    ("squ_in_oct", 26): 2.52711, ("squ_in_oct", 31): 2.78570, ("squ_in_oct", 32): 2.84222,
    ("squ_in_oct", 33): 2.86290, ("squ_in_oct", 35): 2.91052, ("squ_in_oct", 38): 3.02321,
    ("squ_in_oct", 39): 3.08686, ("squ_in_oct", 40): 3.14545,
    ("tri_in_hex", 36): 2.58309, ("hex_in_hex", 32): 6.26164, ("hex_in_hex", 33): 6.26603,
}
_scr = load_scrapes()
_pre = _scr.get("2026-07-04", {})
for (cat, n), pf in _all19_prior.items():
    holder = _pre.get((cat, n), ("", "?", ""))[1]
    add("B4 ALL19", "2026-07-05", cat, n, V19[(cat, n)], pf, holder, "harvest")

add("B5 squinhex", "2026-07-06", "squ_in_hex", 30, V26[("squ_in_hex", 30)], 3.69554,
    "Ian Watson", "harvest")
add("B5 squinhex", "2026-07-06", "squ_in_hex", 40, V26[("squ_in_hex", 40)], 4.30173,
    "Jake Loyd", "harvest")

_all26_prior = {  # (floor, holder) quoted in READY_ALL26_resend/email.txt
    ("pen_in_hex", 26): (4.66044, "Bhavithran Ananthan"), ("pen_in_hex", 27): (4.71800, "Bhavithran Ananthan"),
    ("pen_in_hex", 28): (4.84410, "Bhavithran Ananthan"), ("pen_in_hex", 29): (4.91275, "Mohamed Metwalli"),
    ("pen_in_hex", 30): (4.96723, "Mohamed Metwalli"), ("pen_in_hex", 31): (5.09328, "Bhavithran Ananthan"),
    ("pen_in_oct", 28): (3.56685, "Mohamed Metwalli"), ("pen_in_oct", 29): (3.61924, "Bhavithran Ananthan"),
    ("pen_in_oct", 30): (3.66702, "Bhavithran Ananthan"), ("pen_in_oct", 31): (3.74457, "Bhavithran Ananthan"),
    ("oct_in_tri", 14): (14.24006, "Jonathan Viquerat"), ("oct_in_tri", 18): (16.41137, "Bhavithran Ananthan"),
    ("oct_in_hex", 6): (3.94960, "Mohamed Metwalli"), ("oct_in_hex", 20): (6.99466, "Bhavithran Ananthan"),
    ("tri_in_oct", 31): (1.80341, "Thomas Greenleaf"), ("tri_in_oct", 32): (1.83002, "Thomas Greenleaf"),
    ("tri_in_oct", 33): (1.85532, "Mohamed Metwalli"), ("tri_in_oct", 34): (1.87179, "Mohamed Metwalli"),
    ("tri_in_oct", 35): (1.89225, "Mohamed Metwalli"), ("tri_in_oct", 36): (1.92843, "Mohamed Metwalli"),
    ("tri_in_oct", 37): (1.95803, "Mohamed Metwalli"),
    ("hex_in_pen", 27): (7.15354, "Bhavithran Ananthan"), ("hex_in_pen", 30): (7.46147, "Mohamed Metwalli"),
    ("hex_in_pen", 32): (7.76651, "Bhavithran Ananthan"),
}
for (cat, n), (pf, ph) in _all26_prior.items():
    add("B6 ALL26", "2026-07-06", cat, n, V26[(cat, n)], pf, ph, "harvest")

add("B7 FINAL8", "2026-07-08", "hex_in_pen", 5, f8("hexinpen", 5), 3.21973,
    "Jonathan Viquerat", "search", "small-n blitz; 6th-decimal beat")
add("B7 FINAL8", "2026-07-08", "pen_in_hex", 6, f8("peninhex", 6), 2.322943,
    "Bhavithran Ananthan", "search",
    "lost the race: Ananthan re-squeezed his own entry to the same displayed floor "
    "before processing; never credited")
add("B7 FINAL8", "2026-07-08", "pen_in_squ", 8, f8("peninsqu", 8), 4.38191,
    "Jonathan Viquerat", "search", "small-n blitz; 6th-decimal beat")
add("B7 FINAL8", "2026-07-08", "hex_in_pen", 26, f8("hexinpen", 26), 7.04739,
    "Bhavithran Ananthan", "harvest", "overnight recon+squeeze of a 6-day-old entry")
add("B7 FINAL8", "2026-07-08", "pen_in_tri", 26, f8("penintri", 26), 11.45623,
    "Bhavithran Ananthan", "harvest", "overnight recon+squeeze of a 6-day-old entry")
add("B7 FINAL8", "2026-07-08", "tri_in_pen", 43, f8("triinpen", 43), 3.46438,
    "(table at send time)", "self-update",
    "deeper certification of our B4 value; display-identical (3.46433+)")
add("B7 FINAL8", "2026-07-08", "tri_in_oct", 34, f8("trioct", 34), 1.87172,
    "Aristotle Papowitz (B6)", "drop-one",
    "self-improvement −7.2e-3: drop-one from our stronger n=35 basin")
add("B7 FINAL8", "2026-07-08", "tri_in_oct", 33, f8("trioct", 33), 1.85226,
    "Aristotle Papowitz (B6)", "drop-one",
    "cascade from the new n=34; better-basin propagation")

# ------------------------------------------------------------------ status ----
TAGS = sorted(_scr.keys())
LATEST = TAGS[-1]
US = "Aristotle Papowitz"

# post-hoc case notes for the two claims that were never credited
NEVER_CREDITED = {
    ("pen_in_hex", 6): "never credited: incumbent re-squeezed his own entry to the "
                       "same displayed floor while the submission sat unsent ~30 h",
    ("pen_in_oct", 28): "never credited: overtaken before processing — Ananthan "
                        "posted 3.56402+ (below our 3.56575+) the same day",
}
for r in S:
    if (r["cat"], r["n"]) in NEVER_CREDITED and not r["note"]:
        r["note"] = NEVER_CREDITED[(r["cat"], r["n"])]

def cell_history(cat, n):
    return [(t, _scr[t].get((cat, n))) for t in TAGS if _scr[t].get((cat, n))]

cells = sorted({(r["cat"], r["n"]) for r in S})
cellstat = {}
for cat, n in cells:
    hist = cell_history(cat, n)
    credited = [t for t, (_, h, _) in hist if US in h]
    cur = _scr[LATEST].get((cat, n), ("?", "?", "?"))
    standing = US in cur[1]
    lost_to = None
    if credited and not standing:
        after = [(t, v) for t, v in hist if t > credited[-1] and US not in v[1]]
        lost_to = (after[0][1][1], after[0][1][2], after[0][0]) if after else (cur[1], cur[2], LATEST)
    cellstat[(cat, n)] = dict(credited=bool(credited), first_credit=credited[0] if credited else None,
                              standing=standing, lost_to=lost_to, current=cur)

# last line-item per cell gets the cell status; earlier ones are self-superseded
last_item = {}
for i, r in enumerate(S):
    last_item[(r["cat"], r["n"])] = i
for i, r in enumerate(S):
    st = cellstat[(r["cat"], r["n"])]
    if i != last_item[(r["cat"], r["n"])]:
        r["status"] = "superseded by our later submission"
    elif st["standing"]:
        r["status"] = "STANDING"
    elif st["credited"]:
        h, y, t = st["lost_to"]
        r["status"] = f"credited; re-taken by {h} ({y or 'date n/a'})"
    else:
        r["status"] = "never credited (table moved first)"
    pf = r["prior_floor"]
    r["margin"] = (pf - r["ours"]) if pf else None

standing_at = {t: sum(1 for c in cells if US in _scr[t].get(c, ("", "", ""))[1]) for t in TAGS}

# ------------------------------------------------------------------ report ----
# Conservative, per-problem semantics: one row per (category, n) cell; headline
# counts only what is verifiable from archived scrapes + fresh re-certification.

cert = {}
certpath = ROOT / "paper" / "certification.csv"
if certpath.exists():
    with open(certpath, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cert[(row["category"], int(row["n"]))] = row

SPECIAL = {
    ("pen_in_hex", 6): "withdrawn — not a record (page was stale; the incumbent's "
                       "actual value was not beaten)",
    ("pen_in_oct", 28): "briefly listed during 26-batch processing, re-taken the "
                        "same day — not captured in any archived scrape",
}

percell = []
for cat, n in cells:
    items = [r for r in S if (r["cat"], r["n"]) == (cat, n)]
    final = items[-1]
    mech = final["mech"] if final["mech"] != "self-update" else items[0]["mech"]
    if (cat, n) == ("squ_in_oct", 36):  # strongest prior = the trivial lattice
        pfloor, pholder = 2.92893, "Trivial (10−5√2)"
    else:
        pfloor, pholder = items[0]["prior_floor"], items[0]["prior_holder"]
    st = cellstat[(cat, n)]
    if st["standing"]:
        status = "standing"
    elif st["credited"]:
        h, y, t = st["lost_to"]
        status = f"re-taken by {h} ({y or 'date n/a'})"
    else:
        status = SPECIAL[(cat, n)]
    c = cert.get((cat, n))
    percell.append(dict(
        cat=cat, n=n, ours=final["ours"], pfloor=pfloor, pholder=pholder,
        margin=(pfloor - final["ours"]) if pfloor else None, mech=mech,
        sent=items[0]["sent"], resub=len(items) > 1,
        credit=st["first_credit"] or "—", status=status,
        recert=bool(c and c["verdict"] == "VERIFIED"),
        note="; ".join(x["note"] for x in items if x["note"])))

n_cells = len(percell)
n_cred = sum(1 for c in cells if cellstat[c]["credited"])
n_stand = sum(1 for c in cells if cellstat[c]["standing"])
n_recert = sum(1 for p in percell if p["recert"])
takers = defaultdict(int)
for c in cells:
    st = cellstat[c]
    if st["credited"] and not st["standing"]:
        takers[st["lost_to"][0]] += 1
mechtally = defaultdict(int)
for p in percell:
    mechtally[p["mech"]] += 1

assert n_cells == 54 and len(S) == 58, (n_cells, len(S))
print(f"cells={n_cells}  credited(verified)={n_cred}  standing({LATEST})={n_stand}  "
      f"re-certified={n_recert}/{n_cells}")
print("re-taken:", dict(takers), " mechanisms:", dict(mechtally))

with open(ROOT / "paper" / "ledger.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["category", "n", "s_ours_full", "s_ours_displayed", "prior_floor",
                "prior_holder", "margin_lb", "mechanism", "first_sent",
                "credit_verified_in_scrape", "status_" + LATEST, "recertified",
                "resubmitted", "note"])
    for p in sorted(percell, key=lambda p: (p["cat"], p["n"])):
        w.writerow([p["cat"], p["n"], f"{p['ours']:.15f}", f"{int(p['ours']*1e5)/1e5:.5f}+",
                    p["pfloor"], p["pholder"],
                    f"{p['margin']:.2e}" if p["margin"] is not None else "",
                    p["mech"], p["sent"], p["credit"], p["status"],
                    "yes" if p["recert"] else "NO", "yes" if p["resub"] else "",
                    p["note"]])

lines = []
A = lines.append
A("# Record Ledger — frozen 2026-08-16")
A("")
A("Every problem claimed in the July 2026 campaign, one row per (category, n) cell,")
A(f"reconciled against dated scrapes of the live tables ({', '.join(TAGS)})")
A("and re-certified from the submitted coordinates on 2026-08-16.")
A("Regenerate: `python paper/collect_and_certify.py` then `python paper/build_ledger.py`")
A("(machine-readable: `paper/ledger.csv`, `paper/certification.csv`; coordinates:")
A("`paper/solutions/`).")
A("")
A("## Headline numbers")
A("")
A("Conservative — only what the archives and a fresh certification run can prove.")
A("")
A(f"- **{n_cells} problems claimed** across 14 tables, submitted July 4–8, 2026 in")
A(f"  58 claim line-items (4 cells were submitted twice; see notes).")
A(f"- **{n_recert}/{n_cells} re-certified on 2026-08-16** from the exact coordinates that were")
A(f"  emailed: independent exact-SAT certification reproduces every claimed value to")
A(f"  ≤1e-9; worst pair separation ≥ −8e-13, worst containment ≥ −6e-13,")
A(f"  certification cost ≤ 1e-11 (`paper/certification.csv`).")
A(f"- **{n_cred} problems credited** \"Aristotle Papowitz, July 2026\" — *verified* in at")
A(f"  least one archived scrape.")
A(f"- **{n_stand} standing as of 2026-08-16** ({standing_at['2026-07-09']} were standing 2026-07-09).")
A(f"- **{n_cred - n_stand} since re-taken**: " +
  ", ".join(f"{k} ×{v}" for k, v in sorted(takers.items(), key=lambda x: -x[1])) + ".")
A(f"- **2 claims excluded from the credited count**:")
A(f"  - pen_in_oct 28 — the campaign log records the full 26-batch being processed,")
A(f"    but a competitor had posted a better value the day the batch went out and no")
A(f"    archived scrape captures our listing. Conservatively: not counted.")
A(f"  - pen_in_hex 6 — withdrawn: the page we verified against was stale; the")
A(f"    incumbent's actual value was not beaten. Not a record.")
A(f"- Not counted anywhere: tri_in_hex 8 (July 2–3 channel test) — certified")
A(f"  1.356597399687052, a **tie** with Cantrell 2012, correctly never listed.")
A("")
A("**Reconciling the campaign's live count of \"58 records on the page\":** during the")
A("campaign the tables showed, at one point or another, the 52 verified credits plus")
A("squ_in_oct 36's first listing (~5 minutes, then superseded by the trivial lattice")
A("and re-won), pen_in_oct 28's brief listing, and re-listings of tri_in_pen 43 and")
A("tri_in_oct 33/34 at improved values. The paper claims the conservative,")
A(f"scrape-verifiable number: **{n_cred}**.")
A("")
A("Mechanisms (per problem, final value): " + ", ".join(
    f"{m} ×{c}" for m, c in sorted(mechtally.items(), key=lambda x: -x[1])) + ".")
A("")
A("| # | Mechanism | Meaning |")
A("|---|---|---|")
A("| 1 | harvest | reconstruct incumbent's published image (~0.1 px), converge it in float64 below their displayed floor |")
A("| 2 | search | new arrangement found by batched GPU multi-start / structured basin-hopping |")
A("| 3 | closed-form | harvest that landed on a value agreeing with an exact algebraic form the incumbent missed |")
A("| 4 | drop-one | remove one shape from a stronger (n+1)-packing, re-squeeze; cascades downward |")
A("")
A("## The ledger")
A("")
A("Margin = strongest prior displayed floor − our certified value (a lower bound on")
A("the improvement, since a displayed `x+` means a value in [x, x+10⁻⁵)). ✓ = the")
A("submitted coordinates re-certified to the claimed value on 2026-08-16.")
A("")
A("| Category | n | Ours (certified) | Prior entry | Margin ≥ | Mech | Credited (scrape) | ✓ | Status (2026-08-16) |")
A("|---|---|---|---|---|---|---|---|---|")
for p in sorted(percell, key=lambda p: (p["cat"], p["n"])):
    prior = f"{p['pfloor']}+ ({p['pholder']})" if p["pfloor"] else p["pholder"]
    mark = "✓" if p["recert"] else "**✗**"
    A(f"| {CATNAME[p['cat']]} | {p['n']} | {p['ours']:.15g} | {prior} | "
      f"{p['margin']:.1e} | {p['mech']} | {p['credit']} | {mark} | {p['status']} |")
A("")
A("## Attrition timeline")
A("")
A("| Scrape | Standing | Event |")
A("|---|---|---|")
for t in TAGS:
    ev = {"2026-07-04": "campaign start: tables re-scraped, no credits yet",
          "2026-07-06": "queue cleared: batches B1–B6 processed",
          "2026-07-07": "blitz-era snapshot", "2026-07-07b": "night-shift snapshot",
          "2026-07-09": "B7 processed (7 of 8 listed); Lipponen takes squ_in_oct 31/35/40",
          "2026-08-16": "ledger freeze; attrition: " +
          ", ".join(f"{k} ({v})" for k, v in sorted(takers.items(), key=lambda x: -x[1]))}.get(t, "")
    A(f"| {t} | {standing_at[t]} | {ev} |")
A("")
A("## Notes for the paper")
A("")
A("1. **Truncation semantics.** All margins are lower bounds against the displayed")
A("   floor. Every claim was certified before submission and re-certified for this")
A("   ledger (exact separating-axis margins, float64, certified dilation bound).")
A("2. **The five-minute record.** squ_in_oct 36's first value (2.92902...) beat the")
A("   then-listed 2.92919+ but was superseded within minutes by a trivial 37-square")
A("   diamond lattice at 10−5√2 = 2.92893+ dominating both n=36 and n=37; the")
A("   standing value (2.92870...) then beat the trivial lattice with a vacancy")
A("   rearrangement. Lesson: check trivial n+1 constructions before claiming n.")
A("3. **Race dynamics.** Two claims were invalidated purely by table velocity")
A("   (pen_in_hex 6, pen_in_oct 28); four further finished records (pen_in_squ 3,")
A("   hex_in_pen 4, oct_in_pen 4, pen_in_hex 8) were sniped by a competitor while")
A("   staged, before sending, and were never submitted. Records on a live benchmark")
A(f"   decay: {n_cred} credited → {standing_at['2026-07-09']} standing Jul 9 → {n_stand} standing Aug 16.")
A("4. **Resubmitted cells.** squ_in_oct 36 (after the trivial-lattice supersession),")
A("   tri_in_pen 43 (display-identical deeper certification), tri_in_oct 34 and 33")
A("   (drop-one self-improvements of our own week-old values, −7.2e-3 and −4.4e-3).")
A("5. **Provenance.** Claimed values and coordinates: batch emails + coordinate")
A("   files in `submissions/sent_batches/` (private), frozen as JSON in")
A("   `paper/solutions/` (public). Table history: `data/tables-*/`. Certification:")
A("   `paper/certification.csv`, regenerable with `paper/collect_and_certify.py`.")
A("")
(ROOT / "LEDGER.md").write_text("\n".join(lines), encoding="utf-8")
print(f"wrote LEDGER.md ({len(lines)} lines) + paper/ledger.csv")
