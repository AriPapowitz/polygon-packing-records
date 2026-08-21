"""Tests for the exact-value pipeline: contact extraction and the
residual-validated PSLQ discipline."""
import numpy as np
from mpmath import mp, mpf, pslq, sqrt

from polypack import exact_solve


def test_contact_extraction_squ_in_tri_42(solutions_dir):
    """The n=42 squares-in-triangle record has 39 load-bearing squares
    (the number reported in the companion paper)."""
    sol, n, nsc, vals, S = exact_solve.load(str(solutions_dir / "squ_in_tri_42.json"))
    assert (n, nsc) == (42, 3)
    walls, pairs = exact_solve.f64_contacts(n, nsc, vals, S)
    assert walls and pairs
    active = set(w[0] for w in walls) | set(p[0] for p in pairs) | set(p[2] for p in pairs)
    assert len(active) == 39


def test_pslq_identifies_minimal_polynomial():
    """s = 6 + 8/sqrt(3) satisfies 3 s^2 - 36 s + 44 = 0; PSLQ must find it and
    the relation must survive residual validation."""
    old = mp.dps
    try:
        mp.dps = 60
        s = 6 + 8 / sqrt(3)
        rel = pslq([s**0, s**1, s**2], maxcoeff=10**6, maxsteps=10**5)
        assert rel is not None
        # normalize sign so the leading coefficient is positive
        if rel[2] < 0:
            rel = [-c for c in rel]
        assert rel == [44, -36, 3]
        residual = abs(sum(mpf(c) * s**k for k, c in enumerate(rel)))
        assert residual < mpf(10) ** -50
    finally:
        mp.dps = old


def test_pslq_residual_rejects_near_miss():
    """A value 1e-30 away from the algebraic number must NOT pass residual
    validation, even if PSLQ emits the same candidate relation. This is the
    discipline that killed a spurious degree-6 'closed form' in the campaign."""
    old = mp.dps
    try:
        mp.dps = 60
        s = 6 + 8 / sqrt(3) + mpf(10) ** -30
        rel = pslq([s**0, s**1, s**2], maxcoeff=10**6, maxsteps=10**5)
        if rel is not None:
            residual = abs(sum(mpf(c) * s**k for k, c in enumerate(rel)))
            assert residual > mpf(10) ** -40   # fails the <1e-50 validation gate
    finally:
        mp.dps = old
