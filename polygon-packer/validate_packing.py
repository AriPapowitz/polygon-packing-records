"""Compatibility shim -- the maintained module is polypack.validate_packing.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/validate_packing.py ...      (CLI)
    import validate_packing                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.validate_packing import *  # noqa: E402,F401,F403
from polypack.validate_packing import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
