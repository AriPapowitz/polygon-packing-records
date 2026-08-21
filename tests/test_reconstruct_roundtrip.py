"""Render -> reconstruct -> refine round trip (the harvest instrument,
calibrated end to end on one published packing). Marked slow."""
import numpy as np
import pytest

from polypack import Engine, load_solution
from polypack import reconstruct_gif, render_packing

pytestmark = pytest.mark.slow

KNOWN = 1.356597399687   # tri_in_hex_08 published value


def test_render_reconstruct_refine(solutions_dir, tmp_path):
    src = solutions_dir / "tri_in_hex_08.json"
    png = tmp_path / "packing.png"
    rec = tmp_path / "recon.json"

    render_packing.main([str(src), "-o", str(png), "--no-title"])
    assert png.exists()

    reconstruct_gif.main([str(png), "8", "3", "6", f"{KNOWN:.6f}",
                          "--out", str(rec)])
    assert rec.exists()

    sol, x, S = load_solution(rec)
    eng = Engine(8, 3, 6)
    # the reconstruction is inflated by 0.5% on purpose; give the squeeze
    # enough schedule to traverse that slack (200 x 2e-4 = 4% max shrink)
    S64, x64, valid = eng.refine64(np.asarray(x)[None], np.array([S]),
                                   iters=400, squeeze_rounds=200,
                                   squeeze_step=2e-4)
    assert bool(valid[0])
    s = float(S64[0]) * eng.ratio
    # pixel-scale perturbation + refinement lands back at (or within a hair of)
    # the source basin
    assert abs(s - KNOWN) < 5e-4, f"refined to {s:.9f}, expected ~{KNOWN}"
