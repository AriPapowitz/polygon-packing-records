# n=44 Attack Runbook (4× RTX 4090)

*Prepared 2026-07-04. Target: triangles-in-pentagon n=44, beat 3.52140+ (Derek Wu,
July 2026). A ≤3.51261 packing exists by monotonicity (n=45 record minus one
triangle) — the only strict anomaly on all 12 scraped tables.*

## New tooling (all in `polygon-packer/`, bundled in `runpod/attack_bundle.zip`)

| Tool | Role |
|---|---|
| `pack_core.py` | Shared engine: batched penalty/L-BFGS/anneal/f64-refine, importable, (n,nsi,nsc)-parameterized |
| `packer_bh.py` | **Structured basin-hopping**: elite pool + move portfolio (vacancy teleport, coherent cluster tilt, row shear, local shake, angle scramble, jiggle) + lattice/random immigrants, f32 anneal waves + f64 certification. Smoke-tested: re-finds the tri-in-hex n=8 record basin (1.356597400) in 3 CPU rounds |
| `drop_one.py` | n→n−1 monotonicity construction: batch-refines all n removal choices, saves the best |
| `reconstruct_gif.py` | Site GIF → coordinates: component split + subpixel boundary fits + penalty-balance container fit. n=45 tri-in-pen reconstructed to ~0.15 px (≈0.6%) accuracy |

Local CPU status when this was written: `results/45_recon.json` (reconstruction,
structurally verified), `results/45_recon_refined.json` (valid at 3.5709 — naive
squeeze unjams; the BH kicks are the cure), drop-one scan running.

## GPU assignment (one process per card, `CUDA_VISIBLE_DEVICES=k`)

```bash
# GPU 0 — the main event: n=44 seeded with drop-one candidates + BH
cd results44 && CUDA_VISIBLE_DEVICES=0 python ../packer_bh.py 44 3 5 \
  --batch 2048 --rounds 200 --seed-json "../seeds/44_*.json" \
  --target 3.52139 --seed0 0 | tee bh44_gpu0.log

# GPU 1 — improve n=45 first (better n=45 ⇒ better drop seeds; nailing 3.5127
# would make n=44 ≤ 3.5126 automatic). Re-drop + reseed GPU 0 when it improves.
cd results45 && CUDA_VISIBLE_DEVICES=1 python ../packer_bh.py 45 3 5 \
  --batch 2048 --rounds 200 --seed-json "../seeds/45_recon*.json" \
  --target 3.51262 --seed0 1000 | tee bh45_gpu1.log

# GPU 2 — n=44 independent (no seeds): guards against reconstruction bias
cd results44b && CUDA_VISIBLE_DEVICES=2 python ../packer_bh.py 44 3 5 \
  --batch 2048 --rounds 200 --target 3.52139 --seed0 2000 | tee bh44_gpu2.log

# GPU 3 — secondary target: squares-in-octagon n=36 (2.92919+, near-plateau
# anomaly: n=37 only 7e-5 above ⇒ n=36 likely has slack)
cd results36 && CUDA_VISIBLE_DEVICES=3 python ../packer_bh.py 36 4 8 \
  --batch 2048 --rounds 200 --target 2.92918 --seed0 3000 | tee bh36_gpu3.log
```

Rough scale: one 4090 ran ~127 anneal restarts/min at n=22 last campaign; n=44/45
is ~4× the pair count, so expect ~25–35/min/card — thousands of structured
BH rounds per card per hour. Watch `certified best` in the logs.

## When something beats the table

1. `python validate_packing.py <best>.json --polish --squeeze` (independent
   certification gate — must beat the table value with margin).
2. `python render_packing.py <best>.json`.
3. **Re-check the live table entry that same hour** (it moves daily right now):
   https://erich-friedman.github.io/packing/triinpen/
4. Email erichfriedman68@gmail.com — category, n, claimed s truncated to 6
   digits, coordinates JSON, PNG, method note. Credit: Aristotle Papowitz.
   For n=44 note the method honestly: reconstruction of the published n=45
   packing + removal + structured re-optimization (if that's the winning path).

## Setup on a fresh pod

```bash
unzip attack_bundle.zip -d polygon-packer && cd polygon-packer
pip install -U "jax[cuda12]" numpy scipy matplotlib numba joblib pillow
python -c "import jax; print(jax.devices())"    # expect [CudaDevice(id=0)]
mkdir seeds results44 results44b results45 results36
# regenerate seeds on-pod (or scp the local results/ JSONs into seeds/):
python reconstruct_gif.py results/tri_in_pen_45.gif 45 3 5 3.51261 --out seeds/45_recon.json
# hostile-network fallback (533 B/s pods): see runpod/RUNPOD.md + mss_get.py
```
