"""Compatibility shim -- the maintained module is polypack.render_packing.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/render_packing.py ...      (CLI)
    import render_packing                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.render_packing import *  # noqa: E402,F401,F403
from polypack.render_packing import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
