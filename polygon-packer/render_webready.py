"""Compatibility shim -- the maintained module is polypack.render_webready.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/render_webready.py ...      (CLI)
    import render_webready                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.render_webready import *  # noqa: E402,F401,F403
from polypack.render_webready import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
