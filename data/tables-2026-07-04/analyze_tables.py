"""Analyze scraped Friedman packing tables: anomalies, plateaus, holder ages."""
import csv, glob, math, os, re, sys
from collections import Counter

SCRATCH = os.path.dirname(os.path.abspath(__file__))

def parse_s(raw):
    """Numeric value of a record entry; site format is 'closed form = decimal+'
    or a bare decimal/fraction. Prefer the decimal after the last '='."""
    for cand in reversed(raw.split("=")):
        s = cand.strip().rstrip("+").strip()
        s = s.replace("−", "-").replace(",", "")
        if s.startswith("."):
            s = "0" + s
        expr = re.sub(r"√\(?(\d+(?:\.\d+)?)\)?", r"math.sqrt(\1)", s)
        expr = re.sub(r"(\d)\s*math\.sqrt", r"\1*math.sqrt", expr)  # 2√2 -> 2*sqrt(2)
        try:
            return float(eval(expr, {"math": math, "__builtins__": {}}))
        except Exception:
            continue
    return None

def parse_year(raw):
    m = re.search(r"(19|20)\d\d", str(raw))
    return int(m.group(0)) if m else None

holder_counts = Counter()
for path in sorted(glob.glob(os.path.join(SCRATCH, "*_in_*.csv"))):
    cat = os.path.basename(path)[:-4]
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n = int(r["n"])
            val = parse_s(r["s"])
            yr = parse_year(r["year"])
            rows.append((n, val, r["s"].strip(), r["holder"].strip(), yr))
            if r["holder"].strip():
                holder_counts[r["holder"].strip()] += 1
    rows.sort()
    print(f"\n=== {cat} ({len(rows)} entries) ===")
    bad = [n for n, v, *_ in rows if v is None]
    if bad:
        print(f"  UNPARSED s at n={bad}")
    # anomalies: s strictly decreasing n -> n+1 means entry n has free slack
    by_n = {n: (v, s, h, y) for n, v, s, h, y in rows}
    for n in sorted(by_n):
        if n + 1 in by_n and by_n[n][0] and by_n[n + 1][0]:
            if by_n[n][0] > by_n[n + 1][0] + 1e-9:
                print(f"  ANOMALY: n={n} s={by_n[n][1]} ({by_n[n][2]}, {by_n[n][3]})"
                      f"  >  n={n+1} s={by_n[n+1][1]} ({by_n[n+1][2]}, {by_n[n+1][3]})")
    # plateaus: runs of equal value
    run = []
    for n in sorted(by_n):
        if run and by_n[n][0] is not None and by_n[run[-1]][0] is not None \
           and abs(by_n[n][0] - by_n[run[-1]][0]) < 1e-9 and n == run[-1] + 1:
            run.append(n)
        else:
            if len(run) >= 2:
                print(f"  PLATEAU: n={run[0]}..{run[-1]} s={by_n[run[0]][1]}")
            run = [n]
    if len(run) >= 2:
        print(f"  PLATEAU: n={run[0]}..{run[-1]} s={by_n[run[0]][1]}")
    # age profile
    legacy = [(n, by_n[n][1], by_n[n][2], by_n[n][3]) for n in sorted(by_n)
              if by_n[n][3] and by_n[n][3] <= 2015 and 3 <= n <= 45]
    fresh26 = [n for n in sorted(by_n) if by_n[n][3] == 2026]
    print(f"  legacy (<=2015, 3<=n<=45): {[(n, h, y) for n, s, h, y in legacy]}")
    print(f"  2026 entries: n={fresh26}")

print("\n=== holder counts across scraped categories ===")
for h, c in holder_counts.most_common(20):
    print(f"  {c:4d}  {h}")
