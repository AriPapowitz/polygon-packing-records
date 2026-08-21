"""Compatibility shim -- the maintained module is polypack.drop_one.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/drop_one.py ...      (CLI)
    import drop_one                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.drop_one import *  # noqa: E402,F401,F403
from polypack.drop_one import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
