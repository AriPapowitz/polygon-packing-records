"""Matched-wall-clock CPU comparison: our engine vs the reference packer.

Protocol. Six instances spanning the campaign's regimes. Each system gets the
same wall-clock budget (default 600 s) per instance on the same machine, all
cores, CPU only (the paper reports GPU throughput separately). Our engine:
random multi-start (packer_bh's initialization) through the jitted
shrink-anneal, JIT time counted inside the budget, best survivor refined in
float64 at the end. Reference: Vallejo's polygon_packer.py run in successive
--attempts chunks until the budget is exhausted (chunk size adapts to ~90 s),
best 'Final side length' across chunks. Neither system is seeded or warmed.

Run:  polygon-packer/.venv/Scripts/python paper/baseline_cpu.py [budget_s]
Output: paper/baseline_cpu.csv
"""
import csv, re, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PP = ROOT / "polygon-packer"
PY = str(PP / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(PP))
from pack_core import Engine  # noqa: E402

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0

# (n, nsi, nsc, category, best_known_s, note)
INSTANCES = [
    (8, 3, 6, "tri_in_hex", 1.356597399687, "legacy tie (Cantrell 2012)"),
    (10, 4, 5, "squ_in_pen", 2.793235659, "solver-team entry, documented tie"),
    (12, 3, 5, "tri_in_pen", 1.973299205, "2026 entry, documented tie"),
    (12, 6, 6, "hex_in_hex", 3.941643237, "AlphaEvolve entry, documented tie"),
    (21, 3, 6, "tri_in_hex", 1.998685, "rearrangement control (Lin floor)"),
    (26, 4, 8, "squ_in_oct", 2.527094992259, "our record"),
]

def run_ours(n, nsi, nsc, budget):
    import jax
    t0 = time.time()
    eng = Engine(n, nsi, nsc)
    anneal = eng.make_anneal(1e-8)
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    B = 48
    best_s, best_x, best_S, starts = np.inf, None, None, 0
    while time.time() - t0 < budget:
        xs, Ss = [], []
        for _ in range(B):
            S = eng.lowest_S * rng.uniform(2.0, 4.0)
            xy = rng.uniform(-0.5, 0.5, (n, 2)) * S
            ang = rng.uniform(0, 2 * np.pi, n)
            xs.append(np.column_stack((xy, ang)).ravel())
            Ss.append(S)
        key, sub = jax.random.split(key)
        S_out, x_out = anneal(sub, np.asarray(xs, np.float32),
                              np.asarray(Ss, np.float32))
        S_out = np.asarray(S_out)
        starts += B
        i = int(np.argmin(S_out))
        if S_out[i] * eng.ratio < best_s:
            best_s = float(S_out[i]) * eng.ratio
            best_x, best_S = np.asarray(x_out)[i], float(S_out[i])
    if best_x is not None:
        S_r, _, valid = eng.refine64(best_x[None, :] * 1.001,
                                     np.array([best_S * 1.001]),
                                     iters=300, grow_rounds=60,
                                     squeeze_rounds=120, squeeze_step=1e-4)
        if bool(valid[0]):
            best_s = float(S_r[0]) * eng.ratio
    return best_s, starts, time.time() - t0

def run_reference(n, nsi, nsc, budget):
    t0 = time.time()
    best, attempts, K = np.inf, 0, 8
    with tempfile.TemporaryDirectory() as td:
        while time.time() - t0 < budget - 5:
            left = budget - (time.time() - t0)
            tc = time.time()
            try:
                r = subprocess.run(
                    [PY, str(PP / "polygon_packer.py"), str(n), str(nsi),
                     str(nsc), "--attempts", str(K)],
                    capture_output=True, text=True, cwd=td,
                    timeout=max(30.0, left + 30))
            except subprocess.TimeoutExpired:
                break
            m = re.findall(r"Final side length:\s*([\d.]+)", r.stdout)
            if m:
                best = min(best, float(m[-1]))
                attempts += K
            dt = time.time() - tc
            K = max(4, int(K * min(4.0, max(0.25, 90.0 / max(dt, 1)))))
    return best, attempts, time.time() - t0

rows = []
for n, nsi, nsc, cat, known, note in INSTANCES:
    for name, fn in (("engine", run_ours), ("reference", run_reference)):
        s, starts, dt = fn(n, nsi, nsc, BUDGET)
        gap = s - known
        rows.append([cat, n, name, f"{dt:.0f}", starts, f"{s:.9f}",
                     f"{known:.9f}", f"{gap:+.2e}", note])
        print(f"{cat} n={n} [{name}]: best {s:.9f} vs known {known:.9f} "
              f"({gap:+.1e}), {starts} starts in {dt:.0f}s", flush=True)

with open(ROOT / "paper" / "baseline_cpu.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["category", "n", "system", "seconds", "starts", "best_s",
                "best_known_s", "gap", "note"])
    w.writerows(rows)
print("-> paper/baseline_cpu.csv")
