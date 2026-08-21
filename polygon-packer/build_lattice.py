"""Compatibility shim -- the maintained module is polypack.build_lattice.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/build_lattice.py ...      (CLI)
    import build_lattice                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.build_lattice import *  # noqa: E402,F401,F403
from polypack.build_lattice import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
