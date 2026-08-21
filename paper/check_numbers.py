"""Pre-posting consistency gate: every headline number in the manuscript must
re-derive from the repository's data files. Exits nonzero on any mismatch.

Run:  polygon-packer/.venv/Scripts/python paper/check_numbers.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL = 0


def check(name, got, want):
    global FAIL
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got}" +
          ("" if ok else f"  (paper says {want})"))
    if not ok:
        FAIL = 1


def close(name, got, want, tol):
    global FAIL
    ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got}" +
          ("" if ok else f"  (paper says {want})"))
    if not ok:
        FAIL = 1


led = list(csv.DictReader(open(ROOT / "paper" / "ledger.csv", encoding="utf-8")))
credited = [r for r in led if r["credit_verified_in_scrape"] not in ("", "—")]
standing = [r for r in credited if r["status_2026-08-16"] == "standing"]
check("claims", len(led), 54)
check("credited", len(credited), 52)
check("standing at freeze", len(standing), 41)
check("harvest claims", sum(r["mechanism"] == "harvest" for r in led), 45)
check("harvest credited", sum(r["mechanism"] == "harvest" for r in credited), 44)
margins = sorted(float(r["margin_lb"]) for r in credited)
close("min credited margin", margins[0], 3.9e-7, 1e-8)
close("max credited margin", margins[-1], 7.4e-3, 1e-4)

mx = list(csv.DictReader(open(ROOT / "paper" / "attack_matrix.csv", encoding="utf-8")))
out = Counter(r["outcome"] for r in mx)
check("examined", len(mx), 551)
check("documented ties", out["tie-documented"], 385)
check("beaten (claimed + candidates)",
      out["beaten-claimed"] + out["beaten-candidate"], 59)
check("outcome-unlogged", out["attempted"], 55)

years = {}
for f in (ROOT / "data" / "tables-2026-07-04").glob("*.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8-sig")):
        years[(r["category"], int(r["n"]))] = r.get("year", "")
leg = [r for r in mx
       if (lambda y: y.isdigit() and 1996 <= int(y) <= 2012)(
           years.get((r["category"], int(r["n"])), "").split()[-1]
           if years.get((r["category"], int(r["n"])), "") else "")]
check("legacy examined", len(leg), 39)
check("legacy beaten", sum(r["outcome"].startswith("beaten") for r in leg), 0)
check("legacy documented ties",
      sum(r["outcome"] == "tie-documented" for r in leg), 9)

rt = list(csv.DictReader(open(ROOT / "paper" / "roundtrip.csv", encoding="utf-8")))
check("round-trip cases", len(rt), 55)

hd = [r for r in csv.DictReader(open(ROOT / "paper" / "harvest_displacement.csv",
                                     encoding="utf-8")) if r["max_disp"]]
check("harvest displacement rows compared", len(hd), 43)
big = [r for r in hd if float(r["max_disp_over_side"]) > 0.5]
check("harvest rearrangements (>0.5 side)", len(big), 1)
check("the rearrangement", (big[0]["category"], big[0]["n"]), ("hex_in_pen", "26"))

cert = list(csv.DictReader(open(ROOT / "paper" / "certification.csv", encoding="utf-8")))
check("re-certified packings", len(cert), 55)

sys.exit(FAIL)
