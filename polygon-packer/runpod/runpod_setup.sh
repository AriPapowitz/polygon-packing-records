#!/bin/bash
# One-shot setup for a RunPod GPU pod (any CUDA 12.x template, e.g. "RunPod PyTorch").
# Usage on the pod:   bash runpod_setup.sh
set -e

cd "$(dirname "$0")"

echo "=== Installing JAX (CUDA 12) + deps ==="
pip install -q -U "jax[cuda12]" numpy scipy numba matplotlib

echo "=== GPU check ==="
python -c "import jax; d = jax.devices(); print('devices:', d); assert 'cuda' in str(d[0]).lower() or 'gpu' in str(d[0]).lower(), 'NO GPU FOUND - check the pod template'"

echo "=== Smoke test: 8 triangles in hexagon, batch 256 (expect best ratio ~1.357) ==="
python packer_gpu.py 8 3 6 --batch 256 --waves 1

echo ""
echo "Setup OK. Example hunts:"
echo "  python packer_gpu.py 21 3 6 --batch 4096 --waves 10            # plateau hunt (n=21 control: beat 1.99878)"
echo "  python packer_gpu.py 22 3 6 --batch 4096 --waves 10            # the n=22 record attempt (beat 2.000)"
echo "  python packer_gpu.py 40 3 6 --batch 2048 --waves 5             # fresh large-n block"
echo "On A100/H100 add --x64 for float64 search."
echo "Download the *_gpu_top*.json files and certify locally with validate_packing.py --polish --squeeze."
