"""Closed-form harvest sweep: for every fresh numeric table entry, reconstruct
the GIF, f64-squeeze it, and flag (a) values below the claimed record and
(b) convergence onto exact closed forms (a + b*sqrt(k) family).

Runs standalone on CPU; appends findings to results/sweep_results.md as it goes.
Launch detached:  python closed_form_sweep.py
"""

import csv, itertools, json, os, subprocess, sys, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "tables-2026-07-04")
OUT = os.path.join(HERE, "results", "sweep_results.md")
PY = sys.executable

# category -> (slug, img prefix, nsi, nsc)
CATS = {
    "squ_in_tri": ("squintri", "ts", 4, 3),
    "squ_in_pen": ("squinpen", "", 4, 5),
    "tri_in_pen": ("triinpen", "", 3, 5),
    "tri_in_hex": ("triinhex", "", 3, 6),
    "squ_in_oct": ("squinoct", "", 4, 8),
    "hex_in_hex": ("hexinhex", "", 6, 6),
    "tri_in_squ": ("triinsqu", "t", 3, 4),
    "squ_in_hex": ("squinhex", "", 4, 6),
    "pen_in_pen": ("peninpen", "", 5, 5),
    "pen_in_squ": ("peninsqu", "", 5, 4),
    "pen_in_hex": ("peninhex", "", 5, 6),
    "pen_in_oct": ("peninoct", "", 5, 8),
    "oct_in_tri": ("octintri", "", 8, 3),
    "oct_in_squ": ("octinsqu", "", 8, 4),
    "oct_in_pen": ("octinpen", "", 8, 5),
    "oct_in_hex": ("octinhex", "", 8, 6),
    "tri_in_oct": ("triinoct", "", 3, 8),
    "hex_in_squ": ("hexinsqu", "", 6, 4),
    "hex_in_pen": ("hexinpen", "", 6, 5),
    "hex_in_oct": ("hexinoct", "", 6, 8),
}

def parse_val(raw):
    s = raw.strip().rstrip("+")
    if "=" in s or "√" in s or "sqrt" in s:
        return None                        # already exact — nothing to harvest
    try:
        return float(s)
    except ValueError:
        return None

def closed_forms(v, tol=2e-7):
    """Nearby a+b*sqrt(k)/c forms with small integer parts."""
    hits = []
    for k in (2, 3, 5):
        r = np.sqrt(k)
        for a, b, c in itertools.product(range(0, 15), range(-30, 31), (1, 2, 3, 4, 6)):
            f = a + b * r / c
            if abs(f - v) < tol and abs(b) > 0:
                hits.append(f"{a}+{b}/{c}*sqrt({k}) = {f:.12f}")
    return hits[:2]

def log(line):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

def sweep_entry(cat, slug, prefix, nsi, nsc, n, claimed, holder, year):
    gif = os.path.join(HERE, "results", f"sweep_{cat}_{n}.gif")
    if not os.path.exists(gif):
        ok = False
        for name in (f"{prefix}{n}.gif", f"{n}.gif", f"{n}.png", f"t{n}.gif"):
            r = subprocess.run(["powershell", "-Command",
                f"try {{ Invoke-WebRequest -Uri https://erich-friedman.github.io/packing/{slug}/{name} -OutFile '{gif}' }} catch {{ exit 1 }}"],
                capture_output=True, timeout=60)
            if r.returncode == 0 and os.path.exists(gif) and os.path.getsize(gif) > 500:
                ok = True
                break
        if not ok:
            log(f"- {cat} n={n}: GIF not found, skipped")
            return
    rec = os.path.join(HERE, "results", f"sweep_{cat}_{n}.json")
    r = subprocess.run([PY, os.path.join(HERE, "reconstruct_gif.py"), gif,
                        str(n), str(nsi), str(nsc), f"{claimed:.8f}", "--out", rec],
                       capture_output=True, text=True, timeout=1200)
    if "reconstructed" not in r.stdout:
        log(f"- {cat} n={n}: reconstruction failed")
        return
    import pack_core
    eng = pack_core.Engine(n, nsi, nsc)
    sol, x, S = pack_core.load_solution(rec)
    S64, x64, okv = eng.refine64(x[None], np.array([S]), iters=1200, grow_rounds=60,
                                 grow_rate=1e-4, squeeze_rounds=300, squeeze_step=1e-4)
    if not okv[0]:
        log(f"- {cat} n={n}: refine invalid")
        return
    v = float(S64[0] * eng.ratio)
    forms = closed_forms(v)
    marks = []
    if v < claimed - 1e-5:
        marks.append(f"**RECORD CANDIDATE: {v:.9f} < {claimed}**")
        pack_core.save_solution(os.path.join(HERE, "results", f"sweep_WIN_{cat}_{n}.json"),
                                n, nsi, nsc, float(S64[0]), x64[0])
    if forms:
        marks.append(f"closed form: {forms[0]}")
    tag = "  ".join(marks) if marks else f"refined {v:.9f} (claimed {claimed}) — no find"
    log(f"- {cat} n={n} ({holder} {year}): {tag}")

if __name__ == "__main__":
    only = set(sys.argv[1:])
    log(f"\n## Sweep started (fresh-2026 numeric entries, n=6..45)")
    for cat, (slug, prefix, nsi, nsc) in CATS.items():
        if only and cat not in only:
            continue
        path = os.path.join(DATA, f"{cat}.csv")
        if not os.path.exists(path):
            continue
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            if "2026" not in str(row.get("year", "")):
                continue
            n = int(row["n"])
            if not (6 <= n <= 45):
                continue
            v = parse_val(row["s"])
            if v is None:
                continue
            try:
                sweep_entry(cat, slug, prefix, nsi, nsc, n, v,
                            row["holder"].strip(), row["year"].strip())
            except Exception as e:
                log(f"- {cat} n={n}: error {type(e).__name__}")
    log("## Sweep complete")
