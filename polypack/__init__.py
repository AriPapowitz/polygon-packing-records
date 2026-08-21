"""polypack: certified search, verification, and analysis of packings of unit
regular polygons in regular polygon containers.

The toolkit behind the 2026 record campaign on Erich Friedman's Packing
Center: a batched float32/float64 JAX engine (`pack_core`), an independent
exact separating-axis certifier (`validate_packing`), image-to-coordinates
reconstruction (`reconstruct_gif`), structured basin-hopping and drop-one
record mechanisms (`packer_bh`, `drop_one`), high-precision contact solving
with residual-validated PSLQ (`exact_solve`), and benchmark scrape/diff
tooling (`scrape_tables`).

Common entry points are re-exported lazily so that ``import polypack`` stays
cheap (JAX and numba load only when the machinery is first touched):

    from polypack import Engine, load_solution, certify, build_geometry
"""

__version__ = "1.0.0"

_LAZY = {
    "Engine": "pack_core",
    "load_solution": "pack_core",
    "save_solution": "pack_core",
    "build_geometry": "validate_packing",
    "exact_margins": "validate_packing",
    "pair_separation": "validate_packing",
    "make_penalty": "validate_packing",
    "certify": "validate_packing",
    "polish": "validate_packing",
    "squeeze": "validate_packing",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
