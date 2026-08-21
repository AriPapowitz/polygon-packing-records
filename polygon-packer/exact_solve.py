"""Compatibility shim -- the maintained module is polypack.exact_solve.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/exact_solve.py ...      (CLI)
    import exact_solve                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.exact_solve import *  # noqa: E402,F401,F403
from polypack.exact_solve import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
