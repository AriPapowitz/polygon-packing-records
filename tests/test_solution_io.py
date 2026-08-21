import json

import numpy as np

from polypack import load_solution, save_solution


def test_save_load_roundtrip(tmp_path):
    n, nsi, nsc = 3, 4, 6
    rng = np.random.default_rng(0)
    x = rng.normal(size=n * 3)
    S = 3.25
    path = tmp_path / "sol.json"
    save_solution(path, n, nsi, nsc, S, x, extra={"method": "test"})

    doc = json.loads(path.read_text())
    assert doc["inner_polygons"] == n
    assert doc["method"] == "test"
    ratio = np.sin(np.pi / nsc) / np.sin(np.pi / nsi)
    assert abs(doc["side_length"] - S * ratio) < 1e-15

    sol, x2, S2 = load_solution(path)
    assert S2 == S
    np.testing.assert_allclose(x2, x, atol=0, rtol=0)
