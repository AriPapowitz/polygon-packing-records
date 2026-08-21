"""Compatibility shim -- the maintained module is polypack.packer_bh.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/packer_bh.py ...      (CLI)
    import packer_bh                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.packer_bh import *  # noqa: E402,F401,F403
from polypack.packer_bh import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
