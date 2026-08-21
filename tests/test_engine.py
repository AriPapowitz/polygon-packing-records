"""End-to-end engine tests (CPU, small instances). Marked slow: JIT + search."""
import numpy as np
import pytest

from polypack import Engine, build_geometry, certify, load_solution

pytestmark = pytest.mark.slow


def test_engine_smoke_search():
    """2 unit squares in a square: the anneal must find (near-)optimal s = 2
    from random starts and the result must certify."""
    n, nsi, nsc = 2, 4, 4
    eng = Engine(n, nsi, nsc)
    anneal = eng.make_anneal(3e-8, max_outer=2000)

    import jax
    rng = np.random.default_rng(0)
    B = 8
    xs, Ss = [], []
    for _ in range(B):
        S = eng.lowest_S * rng.uniform(2.0, 4.0)
        xy = rng.uniform(-0.5, 0.5, (n, 2)) * S
        ang = rng.uniform(0, 2 * np.pi, n)
        xs.append(np.column_stack((xy, ang)).ravel())
        Ss.append(S)
    S_out, x_out = anneal(jax.random.PRNGKey(0),
                          np.asarray(xs, np.float32), np.asarray(Ss, np.float32))
    S_out = np.asarray(S_out)
    assert np.isfinite(S_out).any(), "no anneal instance converged"

    i = int(np.argmin(np.where(np.isfinite(S_out), S_out, np.inf)))
    S64, x64, valid = eng.refine64(np.asarray(x_out)[i][None] * 1.001,
                                   np.array([float(S_out[i]) * 1.001]),
                                   iters=300, squeeze_rounds=80)
    assert bool(valid[0])
    rep = certify(x64[0], float(S64[0]), build_geometry(nsi, nsc), nsi, nsc)
    assert rep["worst_pair_separation"] > -1e-11
    assert rep["worst_containment_margin"] > -1e-11
    # optimal is s = 2 (two unit squares side by side)
    assert rep["certified_side_length"] < 2.05


def test_refine64_reconverges_known_basin(solutions_dir):
    """Perturb a published packing at the 1e-3 scale; float64 refinement must
    return to the same basin bottom (the paper's reconstruct-and-squeeze
    premise)."""
    sol, x, S = load_solution(solutions_dir / "tri_in_hex_08.json")
    eng = Engine(8, 3, 6)
    rng = np.random.default_rng(1)
    x_pert = np.asarray(x) + rng.normal(0, 5e-4, len(x))
    S64, x64, valid = eng.refine64(x_pert[None] * 1.002, np.array([S * 1.002]),
                                   iters=300, squeeze_rounds=80)
    assert bool(valid[0])
    s = float(S64[0]) * eng.ratio
    assert abs(s - 1.356597399687) < 1e-5
