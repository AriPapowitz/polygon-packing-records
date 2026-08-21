"""Compatibility shim -- the maintained module is polypack.scan_removals.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/scan_removals.py ...      (CLI)
    import scan_removals                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.scan_removals import *  # noqa: E402,F401,F403
from polypack.scan_removals import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
