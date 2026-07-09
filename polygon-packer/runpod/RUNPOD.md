# Running the GPU packer on RunPod

## Why RunPod (and which GPU)

This workload is a single-GPU, embarrassingly-parallel batch search — RunPod's
community cloud is one of the cheapest venues for exactly this (vast.ai is the
comparable alternative; AWS/GCP cost 3–5x more).

| GPU | ~$/hr (community, check current) | Mode | When |
|---|---|---|---|
| **RTX 4090** | ~$0.35–0.70 | float32 (default) | **Start here.** f32 explores basins fine; precision comes from local CPU polish |
| RTX 5090 | ~$0.9 | float32 | More of the same, faster |
| A100 80GB | ~$1.2–1.9 | `--x64` | Only if f32 results look under-converged |

Billing is per-second; the first benchmark hour costs well under $1.
Use **On-Demand** (not Interruptible/Spot) — an interruption kills in-flight waves.

## Step by step

1. **runpod.io → Deploy → GPU Pod.** Pick RTX 4090 (community cloud), template
   **"RunPod PyTorch 2.x"** (any CUDA 12.x template works — JAX brings its own
   CUDA libraries and only needs the driver). Default disk is fine; no network
   volume needed for a first run.
2. **Upload the bundle.** Open the pod's **JupyterLab** (button on the pod page),
   use the file browser to upload `runpod_bundle.zip` into `/workspace`, then in
   a Jupyter terminal:
   ```bash
   cd /workspace && unzip runpod_bundle.zip -d packer && cd packer
   bash runpod_setup.sh
   ```
   The setup script installs JAX-CUDA, asserts a GPU is visible, and runs a
   smoke test whose expected best ratio is ~1.357 (8 triangles in a hexagon).
3. **Benchmark throughput** (the number that sizes everything):
   ```bash
   python packer_gpu.py 22 3 6 --batch 2048 --waves 2
   ```
   Note restarts/minute vs. the CPU reference (~5 restarts/min/core at n=22).
4. **Hunt.** Priority order from the July 2026 research (see TARGETS.md):
   ```bash
   # control at scale: n=21 — if the GPU finds <1.99878, the method is validated end-to-end
   python packer_gpu.py 21 3 6 --batch 4096 --waves 20 --seed0 0
   # the actual prize: n=22 (any certified <2.000 is an unambiguous record)
   python packer_gpu.py 22 3 6 --batch 4096 --waves 20 --seed0 1000
   # fresh large-n blocks (June-2026 entries, likely unconverged)
   python packer_gpu.py 40 3 6 --batch 2048 --waves 10
   ```
5. **Bring results home.** Download `*_gpu_top*.json` via the Jupyter file
   browser, then locally:
   ```bash
   .venv/Scripts/python.exe validate_packing.py <file> --polish --squeeze
   ```
   Only the locally certified float64 number counts for a record claim.
6. **Stop the pod** when done (Stop ≠ Terminate: Stop keeps the disk billing a
   few cents/hr; Terminate wipes it — fine, since results are downloaded).

## Reading the output

- `wave k: 3900/4096 valid, best ratio = ...` — instances that produced at
  least one valid packing; the best ratio is before CPU polish (expect the
  certified value to improve slightly).
- A wave ends when its **slowest** instance retires; if waves feel long, lower
  `--kicks` (fewer rescue attempts) or `--max-outer`.
- Ratios are in the Packing Center convention: container side / inner side.

## If pip crawls (hostile-host playbook, learned 2026-07-03)

Some community hosts have broken networking: bulk transfers stall at ~500 B/s
(MTU black hole + per-flow token-bucket shaping), while small requests and
Cloudflare bursts look fine. Diagnose with:
```bash
curl -sL --max-time 12 -o /dev/null -w '%{speed_download}\n' \
  'https://files.pythonhosted.org/packages/8f/16/.../jax-0.10.2-py3-none-any.whl'
```
If it reads < 10000 (10 KB/s), don't fight pip. Instead:
1. On a fast machine, resolve the wheel URL list:
   `pip install --dry-run --report r.json --target /tmp/x --platform manylinux2014_x86_64
    --python-version 311 --implementation cp --only-binary=:all: 'jax[cuda12]' scipy`
   then extract `install[].download_info.url` into `wheel_urls.txt`.
2. `scp` `wheel_urls.txt` + `chunked_get.py` (in this bundle) to the pod.
3. `python chunked_get.py --list wheel_urls.txt --dir wheels/ --workers 24 --chunk-mb 2`
   — parallel MSS-clamped range requests surf each connection's fast burst
   (~1.6 MB/s aggregate vs 500 B/s single-flow on the same host).
4. `pip install --no-index --find-links=wheels/ 'jax[cuda12]' scipy`

WARP/proxies do NOT help (the shaping hits tunnels too). Swapping hosts is
cleaner if the pool has alternatives; this playbook is for when it doesn't.

## Costs in perspective

At even 50x CPU throughput, a $0.50 4090-hour ≈ 15,000 n=22 restarts — 100x
tonight's entire CPU campaign. If a few hours of that can't crack a target,
the target needs structured moves, not more brute force.
