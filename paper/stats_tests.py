"""Derivations for the statistical statements quoted in Section 5.

Prints, from the repository's data files alone:
  - the outcome taxonomy of the 551 examined problems;
  - the legacy (1996-2012) sub-audit (join of the attack matrix with the
    campaign-start scrape's year field);
  - where the 55 outcome-unlogged problems fall, per incumbent;
  - two-sided Fisher exact tests on the examined sets for the beaten-rate
    contrasts quoted in the paper (Contributor E vs A; E vs solver team).
    These bound the role of chance only: targeting was preferential, so the
    rates remain coverage numbers, not sampling estimates.

Run:  polygon-packer/.venv/Scripts/python paper/stats_tests.py
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
SCRAPE = ROOT / "data" / "tables-2026-07-04"   # campaign-start snapshot

# paper labels (Table 3): letters by problems examined; mapping is also in
# paper/make_figures.py and reversible from the public attack matrix.
LABEL = {"Mohamed Metwalli": "Contributor E", "Jake Loyd": "Contributor A",
         "Jonathan Viquerat": "Contributor B", "Maurizio Morandi": "Contributor D",
         "Timo Berthold et al": "NLP-solver team",
         "Erich Friedman": "site maintainer"}


def main():
    mx = list(csv.DictReader(open(ROOT / "paper" / "attack_matrix.csv",
                                  encoding="utf-8")))
    print("outcomes:", dict(Counter(r["outcome"] for r in mx)))

    years = {}
    for f in SCRAPE.glob("*.csv"):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            years[(r["category"], int(r["n"]))] = r.get("year", "")
    leg = [r for r in mx
           if (lambda y: y.isdigit() and 1996 <= int(y) <= 2012)(
               years.get((r["category"], int(r["n"])), "").split()[-1]
               if years.get((r["category"], int(r["n"])), "") else "")]
    beaten = sum(r["outcome"].startswith("beaten") for r in leg)
    ties = sum(r["outcome"] == "tie-documented" for r in leg)
    print(f"legacy 1996-2012: examined={len(leg)} beaten={beaten} "
          f"documented ties={ties}")

    att = Counter(LABEL.get(r["incumbent_at_attempt"], r["incumbent_at_attempt"])
                  for r in mx if r["outcome"] == "attempted")
    print("outcome-unlogged (55) per incumbent:", att.most_common(6))

    cover = defaultdict(lambda: [0, 0])
    for r in mx:
        cover[r["incumbent_at_attempt"]][0] += 1
        if r["outcome"].startswith("beaten"):
            cover[r["incumbent_at_attempt"]][1] += 1
    E = cover["Mohamed Metwalli"]
    for name, o in (("Contributor A", cover["Jake Loyd"]),
                    ("NLP-solver team", cover["Timo Berthold et al"])):
        tab = [[E[1], E[0] - E[1]], [o[1], o[0] - o[1]]]
        _, p = fisher_exact(tab, alternative="two-sided")
        print(f"Fisher (two-sided), Contributor E vs {name}: "
              f"{E[1]}/{E[0]} vs {o[1]}/{o[0]}  p = {p:.1e}")


if __name__ == "__main__":
    main()
