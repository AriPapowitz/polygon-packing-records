"""Known-value regression + tamper-detection tests for the certifier.

The reference values are the campaign's published claims; re-deriving them
from the shipped coordinates is the toolkit's core promise (the same check as
paper/collect_and_certify.py, on a fixed subset).
"""
import numpy as np
import pytest

from polypack import build_geometry, certify, load_solution

# (file, nsi, nsc, claimed side ratio)
KNOWN = [
    ("tri_in_hex_08.json", 3, 6, 1.356597399687),      # legacy tie (2012 entry)
    ("squ_in_oct_26.json", 4, 8, 2.527094992258591),   # record, no closed form found
    ("pen_in_squ_08.json", 5, 4, 4.381909606459235),   # record won at the 7th decimal
    ("squ_in_tri_42.json", 4, 3, 10.618802157230435),  # record, s = 6 + 8/sqrt(3)
]


@pytest.mark.parametrize("fname,nsi,nsc,claimed", KNOWN)
def test_known_value_regression(solutions_dir, fname, nsi, nsc, claimed):
    sol, x, S = load_solution(solutions_dir / fname)
    assert sol["inner_sides"] == nsi and sol["container_sides"] == nsc
    rep = certify(np.asarray(x), S, build_geometry(nsi, nsc), nsi, nsc)
    # a valid packing: no negative margins beyond float noise
    assert rep["worst_pair_separation"] > -1e-11
    assert rep["worst_containment_margin"] > -1e-11
    assert rep["centers_inside"]
    # certification costs nothing on a genuinely valid packing
    assert rep["certified_penalty_inflation"] < 1e-10
    # and the certified value reproduces the published claim
    assert abs(rep["certified_side_length"] - claimed) < 1e-9


def test_closed_form_agreement(solutions_dir):
    """The squares-in-triangle records sit just above their exact algebraic
    values (identified by exact_solve at 150+ digits)."""
    for fname, exact in [
        ("squ_in_tri_42.json", 6 + 8 / np.sqrt(3.0)),
        ("squ_in_tri_43.json", 5 + 10 / np.sqrt(3.0)),
    ]:
        sol, x, S = load_solution(solutions_dir / fname)
        rep = certify(np.asarray(x), S, build_geometry(4, 3), 4, 3)
        gap = rep["certified_side_length"] - exact
        assert 0 <= gap < 1e-8, f"{fname}: gap to closed form {gap:+.2e}"


def test_certifier_flags_overlap(solutions_dir):
    sol, x, S = load_solution(solutions_dir / "tri_in_hex_08.json")
    x = np.asarray(x).reshape(-1, 3).copy()
    x[0, :2] = 0.7 * x[0, :2] + 0.3 * x[1, :2]        # shove shape 0 into shape 1
    rep = certify(x.ravel(), S, build_geometry(3, 6), 3, 6)
    assert rep["worst_pair_separation"] < -1e-3
    # the certified (provably valid) size must absorb the violation
    assert rep["certified_side_length"] - rep["raw_side_length"] > 1e-4


def test_certifier_flags_containment(solutions_dir):
    sol, x, S = load_solution(solutions_dir / "tri_in_hex_08.json")
    x = np.asarray(x).reshape(-1, 3).copy()
    x[:, :2] *= 1.2                                    # push everything outward
    rep = certify(x.ravel(), S, build_geometry(3, 6), 3, 6)
    assert rep["worst_containment_margin"] < -1e-3
    assert rep["certified_side_length"] > rep["raw_side_length"]
