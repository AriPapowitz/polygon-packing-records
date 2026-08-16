"""Collect every submitted packing from the sent artifacts and re-certify it.

For each claimed cell, this script:
  1. rebuilds the solution JSON from the artifact that was actually emailed
     (coordinate .txt files / inline email blocks / FINAL8 .json files),
  2. converts inner-side=1 coordinates to the validator's inner-circumradius=1
     convention where needed, cross-checking that R reproduces the claimed s
     to <1e-8 (a wrong unit conversion cannot pass this gate),
  3. runs the independent certifier (exact SAT margins + certified size bound)
     on the submitted coordinates, with no polish or squeeze,
  4. writes paper/solutions/<cat>_<n>.json and paper/certification.csv.

A cell is VERIFIED only if the fresh certified size still equals the claimed
value to <=1e-9. Anything else is reported loudly.

Run:  polygon-packer/.venv/Scripts/python paper/collect_and_certify.py
"""
import csv, json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PP = ROOT / "polygon-packer"
SENT = ROOT / "submissions" / "sent_batches"
OUT = ROOT / "paper" / "solutions"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(PP))
from validate_packing import build_geometry, certify  # noqa: E402

SLUG2CAT = {
    "triinpen": ("tri_in_pen", 3, 5), "triinhex": ("tri_in_hex", 3, 6),
    "triinoct": ("tri_in_oct", 3, 8), "trioct": ("tri_in_oct", 3, 8),
    "squintri": ("squ_in_tri", 4, 3), "squinhex": ("squ_in_hex", 4, 6),
    "squinoct": ("squ_in_oct", 4, 8),
    "penintri": ("pen_in_tri", 5, 3), "peninsqu": ("pen_in_squ", 5, 4),
    "peninhex": ("pen_in_hex", 5, 6), "peninoct": ("pen_in_oct", 5, 8),
    "hexinpen": ("hex_in_pen", 6, 5), "hexinhex": ("hex_in_hex", 6, 6),
    "octintri": ("oct_in_tri", 8, 3), "octinhex": ("oct_in_hex", 8, 6),
}

def s_of(R_circ, nsi, nsc):
    """Side ratio s from container circumradius in inner-circumradius=1 units."""
    return R_circ * math.sin(math.pi / nsc) / math.sin(math.pi / nsi)

def to_circ_units(R_side, rows, nsi):
    """inner-side=1 -> inner-circumradius=1: divide lengths by circumradius(side=1)."""
    c = 1.0 / (2.0 * math.sin(math.pi / nsi))
    return R_side / c, [(x / c, y / c, t) for x, y, t in rows]

# ------------------------------------------------------------- txt parsing ----
HDR = re.compile(r"---\s+(\w+)\s+n=(\d+)\s+s = ([\d.]+)\s+(?:container circumradius )?R = ([\d.]+)")
ROW = re.compile(r"^\s*\d+\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")

def parse_txt(path):
    """Yield (slug, n, s_claimed, R_side_units, rows[x,y,theta])."""
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = HDR.match(line.strip())
        if m:
            if cur and cur[4]:
                yield cur
            cur = [m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), []]
            continue
        r = ROW.match(line)
        if r and cur is not None:
            cur[4].append(tuple(float(g) for g in r.groups()))
    if cur and cur[4]:
        yield cur

def parse_squintri_email(path):
    """The 41/42/43 blocks: '=== n=41   (R = ...) ===' + coordinate rows."""
    claimed = {41: 10.618802157230435, 42: 10.618802157230435, 43: 10.773502698922991}
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"=== n=(\d+)\s+\(R = ([\d.]+)\)", line.strip())
        if m:
            if cur and cur[4]:
                yield cur
            n = int(m.group(1))
            cur = ["squintri", n, claimed[n], float(m.group(2)), []]
            continue
        r = ROW.match(line)
        if r and cur is not None:
            cur[4].append(tuple(float(g) for g in r.groups()))
    if cur and cur[4]:
        yield cur

# ------------------------------------------------------------ collect cells ----
cells = {}  # (cat, n) -> dict(nsi, nsc, R_circ, rows_circ, s_claimed, source)

def put(slug, n, s_claimed, R_side, rows, source, units="side"):
    cat, nsi, nsc = SLUG2CAT[slug]
    if units == "side":
        R, rc = to_circ_units(R_side, rows, nsi)
    else:
        R, rc = R_side, rows
    err = abs(s_of(R, nsi, nsc) - s_claimed)
    assert err < 1e-8 * max(1, s_claimed), \
        f"unit check FAILED {cat} n={n}: R gives s={s_of(R, nsi, nsc)!r} vs claimed {s_claimed!r}"
    assert len(rc) == n, f"{cat} n={n}: {len(rc)} placements"
    cells[(cat, n)] = dict(nsi=nsi, nsc=nsc, R=R, rows=rc, s_claimed=s_claimed, source=source)

for slug, n, sc, R, rows in parse_txt(SENT / "webready_19batch" / "papowitz_coordinates.txt"):
    put(slug, n, sc, R, rows, "B4 ALL19 coordinates txt")
for slug, n, sc, R, rows in parse_txt(SENT / "READY_ALL26_resend" / "papowitz_coordinates.txt"):
    put(slug, n, sc, R, rows, "B5/B6 ALL26 coordinates txt")
for slug, n, sc, R, rows in parse_squintri_email(SENT / "submission_squintri_41_42_43_email.txt"):
    put(slug, n, sc, R, rows, "B3 squintri email")

def put_json(path, source, expect=None):
    d = json.load(open(path, encoding="utf-8"))
    nsi, nsc, n = d["inner_sides"], d["container_sides"], d["inner_polygons"]
    slug = [k for k, v in SLUG2CAT.items() if v[1] == nsi and v[2] == nsc][0]
    rows = [(p["x"], p["y"], p["angle"]) for p in d["placements"]]
    sc = expect if expect is not None else d["side_length"]
    put(slug, n, sc, d["container_circumradius"], rows, source, units="circ")

for f in sorted(F8.glob("*_coordinates.json") if (F8 := SENT / "READY_FINAL_8records") else []):
    put_json(f, "B7 FINAL8 json")  # overwrites tri_in_pen 43 / tri_in_oct 33/34 with final values
put_json(PP / "results" / "36_record2_polished.json", "B2 results json")
put_json(SENT / "papowitz_8_triangles_in_hexagon.json", "n8 tie json",
         expect=1.356597399687052)

# ------------------------------------------------------------------ certify ----
rows_out = []
fails = []
for (cat, n), c in sorted(cells.items()):
    geom = build_geometry(c["nsi"], c["nsc"])
    values = [v for row in c["rows"] for v in row]
    import numpy as np
    rep = certify(np.array(values, float), c["R"], geom, c["nsi"], c["nsc"])
    s_cert = rep["certified_side_length"]
    ok = abs(s_cert - c["s_claimed"]) <= 1e-9 * max(1, c["s_claimed"])
    verdict = "VERIFIED" if ok else "MISMATCH"
    if not ok:
        fails.append((cat, n, c["s_claimed"], s_cert))
    rows_out.append([cat, n, c["source"], f"{c['s_claimed']:.15f}",
                     f"{rep['raw_side_length']:.15f}", f"{s_cert:.15f}",
                     f"{rep['worst_pair_separation']:+.3e}",
                     f"{rep['worst_containment_margin']:+.3e}",
                     f"{rep['certified_penalty_inflation']:.3e}", verdict])
    json.dump(dict(inner_polygons=n, inner_sides=c["nsi"], container_sides=c["nsc"],
                   container_circumradius=c["R"], side_length=c["s_claimed"],
                   certified_side_length_recheck=s_cert, source=c["source"],
                   placements=[dict(x=x, y=y, angle=t) for x, y, t in c["rows"]]),
              open(OUT / f"{cat}_{n:02d}.json", "w"), indent=1)

with open(ROOT / "paper" / "certification.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["category", "n", "source", "s_claimed", "s_raw_fresh", "s_certified_fresh",
                "worst_pair_sep", "worst_containment", "cert_cost", "verdict"])
    w.writerows(rows_out)

print(f"{len(rows_out)} cells collected -> paper/solutions/ ; certified -> paper/certification.csv")
print(f"VERIFIED: {sum(1 for r in rows_out if r[-1] == 'VERIFIED')}   MISMATCH: {len(fails)}")
for cat, n, cl, ce in fails:
    print(f"  MISMATCH {cat} n={n}: claimed {cl!r} fresh-certified {ce!r} (diff {ce-cl:+.2e})")
