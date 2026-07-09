# GPU rental market scan — 2026-07-04 (3-agent research synthesis)

Workload profile: single-GPU JAX float32 lanes, 16–24GB VRAM, checkpointed per
round (interruption-tolerant), no interconnect/storage/egress needs. The right
metric is **CUDA-core FP32 TFLOPS per $/hr** — NOT tensor TFLOPS.

## Verified FP32 specs (the surprise: datacenter cards are BAD here)

RTX 5090: 104.8 TF (32GB) · L40S: 91.6 · RTX 6000 Ada: 91.1 · RTX 4090: 82.6 ·
H100 SXM: 67 (PCIe ~51) · RTX 5080: 56.3 · RTX 3090: 35.6 · L4: 30.3 · **A100: 19.5**.
A100/H100 buy HBM bandwidth, tensor cores, FP64, NVLink — all useless to us.
A 4090 has 4.2× the FP32 of an A100; a 5090 has 5.4×.

## Value ranking for 8 lanes (July 2026 prices, verified)

| Option | $/hr/GPU | 8 lanes | FP32 TF/$·hr | Catch |
|---|---|---|---|---|
| Salad 4090 (batch) | $0.16 | $1.28 | ~460 | container-only, no SSH, gamer PCs, no interruption notice — needs cloud-checkpoint plumbing |
| Salad 5090 (batch) | $0.25 | $2.00 | ~390 | same |
| Vast 3090 interruptible | ~$0.08 | $0.64 | ~320 | each lane 2.3× slower; shines for BREADTH (many targets) not depth |
| **Vast 5090 on-demand** | **~$0.34** | **$2.72** | **~310** | marketplace host variance — filter reliability >99%; needs CUDA 12.8+ jaxlib (current releases fine) |
| RunPod 4090 community | $0.34 | $2.72 | ~240 | the incumbent; solid |
| RunPod 4090 spot | ~$0.17–0.20 | ~$1.5 | ~450 | console-only pricing; 5s SIGTERM notice |
| RunPod 5090 community | $0.69 | $5.52 | ~152 | overpriced vs Vast |
| Nebius L40S preempt | $0.74 | $5.92 | ~124 | best *datacenter-grade* option, still 2.5× worse |
| GCP L4 (free credits) | ($0.28 spot) | — | ~108 | only worth it because credits are free |
| Lambda / DO / Crusoe / A100 anywhere | $1–4 | — | 16–65 | skip entirely |

## Recommendation

1. **Keep the current RunPod 4×4090 pod** for frontier lanes (warm, reliable,
   mid-hunt). Check whether it's Community ($0.34) or Secure ($0.69) — if
   Secure, recreate as Community and halve the cost.
2. **Expansion lanes: Vast.ai RTX 5090s** — same SSH workflow as RunPod (the
   deploy playbook transfers 1:1), 27% more FP32 per card at the same $0.34,
   32GB VRAM headroom. Interruptible bidding (~$0.24) is fine for us: Vast
   *pauses* rather than kills, and our elite pool persists on disk.
   Filters when renting: reliability > 99%, decent inet_down, CUDA ≥ 12.8.
3. **If we ever want maximum-cheap bulk**: Salad 4090 batch at $0.16 — but it
   requires containerizing packer_bh with object-storage checkpointing and
   gives up SSH. Only worth the ~half-day of plumbing for a long steady-state
   campaign, not for this sprint.
4. **Never rent for this workload**: A100, H100, L4 (at paid rates), L40S,
   anything Lambda/DigitalOcean/Crusoe sells.

Full per-provider details and source URLs in the three agent reports
(session task outputs af014e47..., af26f0a8..., a9c62142...).
