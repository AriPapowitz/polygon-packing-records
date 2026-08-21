"""Compatibility shim -- the maintained module is polypack.pack_core.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/pack_core.py ...      (CLI)
    import pack_core                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.pack_core import *  # noqa: E402,F401,F403

if __name__ == "__main__":
    raise SystemExit("pack_core is a library module; see polypack.pack_core")
