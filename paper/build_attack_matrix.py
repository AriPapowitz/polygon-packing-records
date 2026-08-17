"""Build the campaign attack matrix: every (category, n) cell examined, how,
and with what outcome. This supplies the denominators the audit demanded.

Evidence streams (all local):
  A  polygon-packer/results/sweep_results.md      per-entry harvest sweep log
     (ties with refined-vs-claimed values, record candidates, failures, skips)
  B  results/blitz_targets.txt + blitz_final.txt  small-n search blitz (per-entry)
  C  results/blitz2_targets.txt (+ result JSONs)  n<=8 blitz; per-entry outcomes
     known for the finds/rejects, remainder ties per the campaign aggregate
  D  results/blitz3_targets.txt + snapshot        n=9..12 sweep (per-entry)
  E  results/night_harvest/ file names            drop-one (do0_*) and night-BH
     (bh_*) attempts per cell
  F  results/ root file names                     BH/search/reconstruction
     attempts (<n>_<nsi>_in_<nsc>_bh_*, *_recon, *_hole_*, ...)
  G  curated entries from CAMPAIGN.md             experiments documented in the
     log but not in machine-readable files (marked source=CAMPAIGN)
plus the frozen ledger (paper/ledger.csv) for the 54 claims and their status.

Outcome precedence (strongest wins):
  beaten-claimed > beaten-candidate (found but not submitted / rejected)
  > tie-documented (refined value logged) > no-improvement (aggregate)
  > attempt-failed (recon/refine failed) > skipped (no usable image)

Output: paper/attack_matrix.csv + ATTACK_MATRIX.md (summary for the paper).
Run:  polygon-packer/.venv/Scripts/python paper/build_attack_matrix.py
"""
import csv, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "polygon-packer" / "results"
NH = RES / "night_harvest"

PAIR2CAT = {(3, 4): "tri_in_squ", (3, 3): "tri_in_tri", (3, 5): "tri_in_pen",
            (3, 6): "tri_in_hex", (3, 8): "tri_in_oct",
            (4, 3): "squ_in_tri", (4, 5): "squ_in_pen", (4, 6): "squ_in_hex",
            (4, 8): "squ_in_oct",
            (5, 3): "pen_in_tri", (5, 4): "pen_in_squ", (5, 5): "pen_in_pen",
            (5, 6): "pen_in_hex", (5, 8): "pen_in_oct",
            (6, 3): "hex_in_tri", (6, 4): "hex_in_squ", (6, 5): "hex_in_pen",
            (6, 6): "hex_in_hex", (6, 8): "hex_in_oct",
            (8, 3): "oct_in_tri", (8, 4): "oct_in_squ", (8, 5): "oct_in_pen",
            (8, 6): "oct_in_hex", (8, 8): "oct_in_oct"}

RANK = {"beaten-claimed": 6, "beaten-candidate": 5, "tie-documented": 4,
        "no-improvement": 3, "attempt-failed": 2, "skipped": 1, "attempted": 0}

cells = {}  # (cat,n) -> dict(outcome, mechanisms=set, holder, floor, ours, evidence=set)

def touch(cat, n, outcome, mech, ev, holder=None, floor=None, ours=None):
    key = (cat, int(n))
    c = cells.setdefault(key, dict(outcome="attempted", mechanisms=set(),
                                   holder="", floor="", ours="", evidence=set()))
    c["mechanisms"].add(mech)
    c["evidence"].add(ev)
    if RANK[outcome] > RANK[c["outcome"]]:
        c["outcome"] = outcome
    if holder and not c["holder"]:
        c["holder"] = holder
    if floor and not c["floor"]:
        c["floor"] = str(floor)
    if ours and (not c["ours"] or RANK[outcome] >= 4):
        c["ours"] = str(ours)

# ---- A: harvest sweep log ---------------------------------------------------
sweep = open(RES / "sweep_results.md", encoding="utf-8", errors="replace").read()
for m in re.finditer(
        r"^- (\w+_in_\w+) n=(\d+)(?: \(([^)]*?)\))?: (.+)$", sweep, re.M):
    cat, n, holder, rest = m.group(1), m.group(2), m.group(3) or "", m.group(4)
    holder = re.sub(r"\s+(January|February|March|April|May|June|July|20\d\d).*$", "", holder).strip()
    if "RECORD CANDIDATE" in rest:
        mm = re.search(r"RECORD CANDIDATE: ([\d.]+) < ([\d.]+)", rest)
        touch(cat, n, "beaten-candidate", "harvest", "sweep_results.md",
              holder, mm.group(2), mm.group(1))
    elif rest.startswith("refined"):
        mm = re.search(r"refined ([\d.]+) \(claimed ([\d.]+)\)", rest)
        touch(cat, n, "tie-documented", "harvest", "sweep_results.md",
              holder, mm.group(2), mm.group(1))
    elif rest.startswith("closed form"):
        touch(cat, n, "tie-documented", "harvest (closed-form identified)",
              "sweep_results.md", holder)
    elif "reconstruction failed" in rest or "refine invalid" in rest:
        touch(cat, n, "attempt-failed", "harvest", "sweep_results.md", holder)
    elif "GIF not found" in rest:
        touch(cat, n, "skipped", "harvest", "sweep_results.md", holder)

# ---- B: blitz 1 (small n, per-entry finals) ---------------------------------
tgt1 = {}
for line in open(RES / "blitz_targets.txt", encoding="utf-8"):
    p = line.split()
    if len(p) >= 5:
        # holder column holds first names only -- leave blank, enrich from scrape
        tgt1[(p[0], int(p[1]))] = (p[4], "")
for line in open(RES / "blitz_final.txt", encoding="utf-8"):
    m = re.match(r"(\w+_in_\w+) n=(\d+) claim=([\d.]+) best=([\d.]+)", line)
    if m:
        cat, n, claim, best = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        holder = tgt1.get((cat, n), ("", ""))[1]
        out = "beaten-candidate" if best < claim else "tie-documented"
        touch(cat, n, out, "search (blitz-1)", "blitz_final.txt", holder, claim, best)
for (cat, n), (floor, holder) in tgt1.items():
    touch(cat, n, "no-improvement", "search (blitz-1)", "blitz_targets.txt", holder, floor)

# ---- C: blitz 2 (n<=8) ------------------------------------------------------
for line in open(RES / "blitz2_targets.txt", encoding="utf-8"):
    p = line.split()
    if len(p) >= 5:
        touch(p[0], int(p[1]), "no-improvement", "search (blitz-2)",
              "blitz2_targets.txt (aggregate: 49 ties)", floor=p[4])
# per-entry outcomes documented in the campaign log / result JSONs:
touch("pen_in_hex", 6, "beaten-candidate", "search (blitz-2)", "blitz2_penhex6.json")
touch("pen_in_hex", 8, "beaten-candidate", "search (blitz-2)",
      "blitz2_penhex8.json (sniped while staged, never submitted)")
touch("pen_in_squ", 8, "beaten-candidate", "search (blitz-2)", "blitz2_pensqu8.json")
touch("pen_in_tri", 8, "attempt-failed", "search (blitz-2)",
      "blitz2_pentri8.json (live-gate reject: table moved)")

# ---- D: blitz 3 (n=9..12) ---------------------------------------------------
tgt3 = {}
for line in open(RES / "blitz3_targets.txt", encoding="utf-8"):
    p = line.split()
    if len(p) >= 5:
        tgt3[(p[0], int(p[1]))] = p[4]
for line in open(RES / "blitz3_results_snapshot.txt", encoding="utf-8"):
    m = re.match(r"(\w+_in_\w+) n=(\d+) claim=([\d.]+) best=([\d.]+)", line)
    if m:
        cat, n, claim, best = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        out = "beaten-candidate" if best < claim else "tie-documented"
        touch(cat, n, out, "search (blitz-3)", "blitz3_results_snapshot.txt",
              floor=claim, ours=best)
for (cat, n), floor in tgt3.items():
    touch(cat, n, "no-improvement", "search (blitz-3)",
          "blitz3_targets.txt (aggregate: 0/69 improvements)", floor=floor)

# ---- E: night harvest file names -------------------------------------------
for f in NH.iterdir():
    m = re.match(r"do0_(\w+_in_\w+)_(\d+)_", f.name)
    if m:
        touch(m.group(1), m.group(2), "attempted", "drop-one (night sweep)",
              "night_harvest/do0_*")
        continue
    m = re.match(r"bh_(\w+_in_\w+)_(\d+)_g", f.name)
    if m:
        touch(m.group(1), m.group(2), "attempted", "search (night BH)",
              "night_harvest/bh_*")

# ---- F: results/ root search & reconstruction files -------------------------
for f in RES.iterdir():
    m = re.match(r"(\d+)_(\d)_in_(\d)_", f.name)
    if m and (int(m.group(2)), int(m.group(3))) in PAIR2CAT:
        cat = PAIR2CAT[(int(m.group(2)), int(m.group(3)))]
        touch(cat, m.group(1), "attempted", "search (BH/GPU)", "results/*_bh_*")

# ---- G: curated documented experiments from CAMPAIGN.md ---------------------
CAMP = [  # (cat, n, outcome, mech, note)
    ("tri_in_hex", 8, "tie-documented", "search+submit", "certified tie with Cantrell 2012 (channel test)"),
    ("tri_in_hex", 16, "tie-documented", "search", "refined to record value +1e-9"),
    ("tri_in_hex", 17, "tie-documented", "search", "refined to record value +1e-9"),
    ("tri_in_hex", 18, "tie-documented", "search", "refined to record value +1e-9"),
    ("tri_in_hex", 21, "no-improvement", "search (control)", "4,700+ starts, 400+ BH rounds; stuck 1.99877 vs 1.99869+"),
    ("tri_in_hex", 22, "no-improvement", "search (plateau)", "s=2 plateau rigid"),
    ("tri_in_hex", 23, "no-improvement", "search (plateau)", "s=2 plateau rigid"),
    ("tri_in_hex", 24, "no-improvement", "search (plateau)", "s=2 plateau rigid"),
    ("tri_in_hex", 28, "no-improvement", "search (fresh tail)", "best 2.33299 vs 2.32251+, miss 1.05e-2"),
    ("squ_in_pen", 8, "tie-documented", "search", "reproduced incumbent to 7 digits"),
    ("squ_in_pen", 10, "tie-documented", "search", "reproduced incumbent to 7 digits"),
    ("squ_in_pen", 14, "tie-documented", "search", "reproduced incumbent to 7 digits"),
    ("squ_in_pen", 15, "tie-documented", "search", "reproduced incumbent to 7 digits"),
    ("squ_in_pen", 24, "no-improvement", "search (lane)", "overnight lane, 4.4e-4 from Loyd, no find"),
    ("squ_in_tri", 16, "tie-documented", "search (lane)", "documented tie"),
    ("squ_in_tri", 20, "no-improvement", "search (lane)", "chased Lin at +1.8e-5, no find"),
    ("squ_in_tri", 29, "tie-documented", "search", "documented tie"),
    ("squ_in_tri", 38, "no-improvement", "search (lane)", "Costantino sub-form target, no find"),
    ("squ_in_tri", 44, "no-improvement", "search", "Wu basin exhausted"),
    ("squ_in_oct", 16, "no-improvement", "search (plateau)", "5sqrt2-5 plateau"),
    ("oct_in_squ", 15, "no-improvement", "search (plateau)", "4+4sqrt2 plateau; f32 sightings were violations"),
    ("oct_in_oct", 24, "tie-documented", "search (night BH)", "tie +8.4e-6 above Ananthan floor"),
    ("oct_in_hex", 18, "tie-documented", "search (night BH)", "tie +4.5e-6 above Ananthan floor"),
    ("tri_in_pen", 45, "attempt-failed", "harvest (recon)", "3 attempts land ~3.53 vs 3.5126; below resolution floor"),
]
for cat, n, out, mech, note in CAMP:
    touch(cat, n, out, mech, f"CAMPAIGN.md: {note}")

# ---- ledger: the 54 claims override -----------------------------------------
led = list(csv.DictReader(open(ROOT / "paper" / "ledger.csv", encoding="utf-8")))
for r in led:
    status = r["status_2026-08-16"]
    touch(r["category"], r["n"], "beaten-claimed", r["mechanism"],
          f"ledger: {status}", r["prior_holder"], r["prior_floor"], r["s_ours_full"])

# ---- enrich holders from the pre-campaign scrape ----------------------------
pre = {}
for f in (ROOT / "data" / "tables-2026-07-04").glob("*.csv"):
    for row in csv.DictReader(open(f, encoding="utf-8", errors="replace")):
        ns = [int(x) for x in re.findall(r"\d+", row["n"])]
        for n in (range(ns[0], ns[-1] + 1) if len(ns) > 1 else ns):
            pre[(row["category"], n)] = row["holder"].strip()
for key, c in cells.items():
    if not c["holder"]:
        c["holder"] = pre.get(key, "")

# ---- write ------------------------------------------------------------------
with open(ROOT / "paper" / "attack_matrix.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["category", "n", "outcome", "mechanisms", "incumbent_at_attempt",
                "incumbent_floor", "our_best_logged", "evidence"])
    for (cat, n), c in sorted(cells.items()):
        w.writerow([cat, n, c["outcome"], "; ".join(sorted(c["mechanisms"])),
                    c["holder"], c["floor"], c["ours"],
                    " | ".join(sorted(c["evidence"]))])

# ---- summaries --------------------------------------------------------------
tot = len(cells)
by_out = defaultdict(int)
for c in cells.values():
    by_out[c["outcome"]] += 1
hold = defaultdict(lambda: [0, 0, 0])  # examined, beaten(any), tied
for (cat, n), c in sorted(cells.items()):
    h = c["holder"] or "?"
    if h.startswith("Trivial"):
        h = "Trivial"
    hold[h][0] += 1
    if c["outcome"].startswith("beaten"):
        hold[h][1] += 1
    elif c["outcome"] == "tie-documented":
        hold[h][2] += 1

lines = []
A = lines.append
A("# Attack matrix — what the campaign examined, and with what outcome")
A("")
A("Reconstructed 2026-08-17 from local evidence (sweep logs, blitz target and")
A("result files, night-harvest artifacts, campaign log, frozen ledger).")
A("Machine-readable: `paper/attack_matrix.csv`. Regenerate:")
A("`python paper/build_attack_matrix.py`.")
A("")
A("## Denominators")
A("")
A(f"- **{tot} distinct (category, n) problems examined** by at least one mechanism.")
for out in ["beaten-claimed", "beaten-candidate", "tie-documented",
            "no-improvement", "attempt-failed", "skipped", "attempted"]:
    if by_out.get(out):
        desc = {"beaten-claimed": "beaten and claimed (the ledger's 54)",
                "beaten-candidate": "beaten but never claimed (sniped, rejected, or superseded before submission)",
                "tie-documented": "documented ties (our refined value logged, no improvement possible in basin)",
                "no-improvement": "searched with no improvement (per-entry or aggregate-logged)",
                "attempt-failed": "attempt failed (reconstruction/refine failure or live-gate reject)",
                "skipped": "skipped (no usable image)",
                "attempted": "attempted, outcome not logged"}[out]
        A(f"- **{by_out[out]}** {desc}.")
A("")
A("## Examined vs beaten, by incumbent at time of attempt")
A("")
A("| Incumbent | Examined | Beaten | Documented ties |")
A("|---|---|---|---|")
for h, (ex, be, ti) in sorted(hold.items(), key=lambda x: (-x[1][0], x[0])):
    if ex >= 3 and h not in ("?", ""):
        A(f"| {h} | {ex} | {be} | {ti} |")
A("")
A("Caveats: targeting was not uniform (fresh, suspected-loose entries were")
A("prioritized; some blocks were deliberately avoided as actively swept), so")
A("these are audit coverage numbers, not unbiased sampling rates. Rows marked")
A("'attempted, outcome not logged' had compute spent but no per-entry outcome")
A("recorded; aggregate-only outcomes are marked in the evidence column.")
(ROOT / "ATTACK_MATRIX.md").write_text("\n".join(lines), encoding="utf-8")
print(f"{tot} cells -> paper/attack_matrix.csv + ATTACK_MATRIX.md")
print(dict(by_out))
