"""Analytic unit tests for the exact separating-axis geometry."""
import numpy as np

from polypack.build_lattice import S as LATTICE_S, lattice_placements
from polypack.validate_packing import (
    build_geometry,
    exact_margins,
    pair_separation,
    transform_all,
)

SQRT2_OVER_2 = np.sqrt(2.0) / 2.0


def _diamond_pair(distance):
    """Two unit-circumradius squares (vertices on the axes), centers on the
    x-axis `distance` apart, no rotation."""
    geom = build_geometry(4, 4)
    values = np.array([0.0, 0.0, 0.0, distance, 0.0, 0.0])
    verts, normals = transform_all(values, geom["inner_vertices"], geom["inner_normals"])
    return verts, normals


def test_pair_separation_disjoint():
    verts, normals = _diamond_pair(3.0)
    # best separating axis is an edge normal at 45 degrees:
    # gap = 3*cos(45) - 2*apothem = sqrt(2)/2
    sep = pair_separation(verts[0], verts[1], normals[0], normals[1])
    assert abs(sep - SQRT2_OVER_2) < 1e-12


def test_pair_separation_penetrating():
    verts, normals = _diamond_pair(1.0)
    sep = pair_separation(verts[0], verts[1], normals[0], normals[1])
    assert abs(sep + SQRT2_OVER_2) < 1e-12          # penetration depth sqrt(2)/2


def test_containment_margin_single_shape():
    geom = build_geometry(4, 4)
    values = np.array([0.0, 0.0, 0.0])
    # container square with circumradius 2: apothem sqrt(2); the diamond's
    # vertices project to sqrt(2)/2 on the container normals
    worst_pair, worst_containment, _, centers_inside = exact_margins(values, 2.0, geom)
    assert worst_pair == np.inf                     # no pairs with n = 1
    assert abs(worst_containment - SQRT2_OVER_2) < 1e-12
    assert centers_inside


def test_lattice_is_valid_and_side_two():
    """The exact 24-triangle tiling of the side-2 hexagon: every margin is
    zero up to float rounding and the side ratio is 2."""
    placements = lattice_placements()
    assert len(placements) == 24
    values = np.array(placements, float).ravel()
    geom = build_geometry(3, 6)
    worst_pair, worst_containment, touching, centers_inside = exact_margins(
        values, LATTICE_S, geom)
    assert worst_pair > -1e-9
    assert worst_containment > -1e-9
    assert touching > 0                             # it is a tiling: contacts everywhere
    assert centers_inside
    ratio = LATTICE_S * np.sin(np.pi / 6) / np.sin(np.pi / 3)
    assert abs(ratio - 2.0) < 1e-12
