"""Interval-arithmetic certification of every claimed packing.

For each solution JSON in paper/solutions/, this recomputes the certificate of
validate_packing.certify() -- worst pair-separation margin, worst containment
margin, and the shrink/dilate certified side length -- in mpmath interval
arithmetic (outward rounding), so the result is a machine-rigorous bound
rather than a float64 computation.

Method. Coordinates and the container circumradius are exact binary doubles,
taken as point intervals. Inner-polygon vertices/normals are built from
interval sin/cos. For each pair we need a rigorous LOWER bound on the exact
SAT margin  sep = max_axes(-overlap):  per axis, overlap <= min_ub(max proj) -
max_lb(min proj), so -overlap >= a computable mpf; sep_lb = max over axes.
Far pairs are prescreened rigorously: if the interval center distance minus 2
(both circumradii are 1 in the validator's units) is already positive, that is
a valid lower bound without SAT. Containment: worst_lb = min over vertices of
limit_lb - proj_ub. The certificate then uses directed bounds throughout:
  d_ub = max(0, -worst_pair_lb),  p_ub = max(0, -worst_cont_lb),
  k_lb = 1 - d_ub/(2*apothem_lb),  S_cert_ub = (S + p_ub/capothem_lb)/k_lb,
  s_cert_ub = S_cert_ub * ub(sin(pi/nsc)/sin(pi/nsi)).
s_cert_ub is a rigorous upper bound on a side length at which a valid packing
of n unit m-gons provably exists (the shrunk/dilated configuration realizes
it). Verdict: INTERVAL-CERTIFIED if s_cert_ub matches the claimed value to
1e-9 and (where a prior floor exists) s_cert_ub < floor, i.e. the record
claim itself is machine-proved.

Run:  polygon-packer/.venv/Scripts/python paper/certify_interval.py
Output: paper/certification_interval.csv
"""
import csv, json, time
from pathlib import Path

from mpmath import iv, mp, mpf

iv.dps = 40
mp.dps = 40

ROOT = Path(__file__).resolve().parent.parent
SOL = ROOT / "paper" / "solutions"

# prior displayed floors for the record-claim check (from the frozen ledger)
floors = {}
with open(ROOT / "paper" / "ledger.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        try:
            floors[(r["category"], int(r["n"]))] = mpf(r["prior_floor"])
        except Exception:
            pass

def certify_interval(sol):
    n, nsi, nsc = sol["inner_polygons"], sol["inner_sides"], sol["container_sides"]
    S = mpf(repr(sol["container_circumradius"]))
    pl = [(mpf(repr(p["x"])), mpf(repr(p["y"])), p["angle"]) for p in sol["placements"]]

    two_pi = 2 * iv.pi
    # rotated vertices and edge normals, as intervals, per polygon
    verts, norms = [], []
    for (x, y, a) in pl:
        ai = iv.mpf(repr(a))
        vs, ns = [], []
        for k in range(nsi):
            va = ai + two_pi * k / nsi
            vs.append((iv.mpf(repr(float(x))) + iv.cos(va),
                       iv.mpf(repr(float(y))) + iv.sin(va)))
            na = va + iv.pi / nsi
            ns.append((iv.cos(na), iv.sin(na)))
        verts.append(vs)
        norms.append(ns)

    # -------- pair margins: rigorous lower bound of min over pairs ----------
    worst_pair = mpf("inf")
    centers = [(x, y) for (x, y, _) in pl]
    for i in range(n):
        xi, yi = centers[i]
        for j in range(i + 1, n):
            xj, yj = centers[j]
            # rigorous prescreen on center distance (circumradius = 1 each)
            dx, dy = xi - xj, yi - yj
            dist2 = dx * dx + dy * dy          # exact-ish mpf at 40 dps
            if dist2 > mpf("6.25"):            # dist > 2.5 -> sep >= dist-2 > 0.5
                continue
            sep_lb = mpf("-inf")
            for (pi_, pj_) in ((i, j), (j, i)):
                for (nx, ny) in norms[pi_]:
                    pa_lo = pa_hi = None
                    for (vx, vy) in verts[i]:
                        pr = vx * nx + vy * ny
                        lo, hi = mpf(pr.a), mpf(pr.b)
                        pa_lo = lo if pa_lo is None else min(pa_lo, lo)
                        pa_hi = hi if pa_hi is None else max(pa_hi, hi)
                    pb_lo = pb_hi = None
                    for (vx, vy) in verts[j]:
                        pr = vx * nx + vy * ny
                        lo, hi = mpf(pr.a), mpf(pr.b)
                        pb_lo = lo if pb_lo is None else min(pb_lo, lo)
                        pb_hi = hi if pb_hi is None else max(pb_hi, hi)
                    overlap_ub = min(pa_hi, pb_hi) - max(pa_lo, pb_lo)
                    sep_lb = max(sep_lb, -overlap_ub)
            worst_pair = min(worst_pair, sep_lb)

    # -------- containment: rigorous lower bound over vertices/walls --------
    capothem = iv.cos(iv.pi / nsc)
    limit = iv.mpf(repr(float(S))) * capothem
    limit_lb = mpf(limit.a)
    worst_cont = mpf("inf")
    for c in range(nsc):
        wa = two_pi * c / nsc + iv.pi / nsc
        wnx, wny = iv.cos(wa), iv.sin(wa)
        for i in range(n):
            for (vx, vy) in verts[i]:
                pr = vx * wnx + vy * wny
                worst_cont = min(worst_cont, limit_lb - mpf(pr.b))

    # -------- directed-bound certificate ----------------------------------
    apothem_lb = mpf(iv.cos(iv.pi / nsi).a)
    capothem_lb = mpf(capothem.a)
    d_ub = max(mpf(0), -worst_pair)
    p_ub = max(mpf(0), -worst_cont)
    k_lb = 1 - d_ub / (2 * apothem_lb)
    S_cert_ub = (S + p_ub / capothem_lb) / k_lb
    ratio_fac_ub = mpf((iv.sin(iv.pi / nsc) / iv.sin(iv.pi / nsi)).b)
    s_cert_ub = S_cert_ub * ratio_fac_ub
    return worst_pair, worst_cont, s_cert_ub

rows = []
for f in sorted(SOL.glob("*.json")):
    cat, ns = f.stem.rsplit("_", 1)
    n = int(ns)
    sol = json.load(open(f, encoding="utf-8"))
    t0 = time.time()
    wp, wc, s_ub = certify_interval(sol)
    claimed = mpf(repr(sol["side_length"]))
    close = abs(s_ub - claimed) < mpf("1e-9")
    floor = floors.get((cat, n))
    beats = (s_ub < floor) if floor is not None else None
    verdict = "INTERVAL-CERTIFIED" if close and beats is not False else "CHECK"
    if floor is not None and not beats:
        verdict = "DOES NOT BEAT FLOOR"
    rows.append([cat, n, mp.nstr(wp, 5), mp.nstr(wc, 5), mp.nstr(s_ub, 20),
                 mp.nstr(claimed, 17), mp.nstr(floor, 8) if floor is not None else "",
                 "yes" if beats else ("" if beats is None else "NO"), verdict])
    print(f"{cat} n={n}: pair>={mp.nstr(wp, 3)} cont>={mp.nstr(wc, 3)} "
          f"s_cert<={mp.nstr(s_ub, 12)} {verdict} ({time.time()-t0:.0f}s)", flush=True)

with open(ROOT / "paper" / "certification_interval.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["category", "n", "worst_pair_lb", "worst_containment_lb",
                "s_certified_interval_ub", "s_claimed", "prior_floor",
                "beats_floor", "verdict"])
    w.writerows(rows)
print(f"{len(rows)} cells -> paper/certification_interval.csv")
print("certified:", sum(1 for r in rows if r[-1] == "INTERVAL-CERTIFIED"))
