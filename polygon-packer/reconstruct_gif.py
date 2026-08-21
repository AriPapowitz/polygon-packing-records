"""Compatibility shim -- the maintained module is polypack.reconstruct_gif.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/reconstruct_gif.py ...      (CLI)
    import reconstruct_gif                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.reconstruct_gif import *  # noqa: E402,F401,F403
from polypack.reconstruct_gif import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
