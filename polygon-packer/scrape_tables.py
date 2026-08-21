"""Compatibility shim -- the maintained module is polypack.scrape_tables.

Kept so campaign-era commands and imports keep working unchanged:
    python polygon-packer/scrape_tables.py ...      (CLI)
    import scrape_tables                            (with polygon-packer/ on sys.path)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from polypack.scrape_tables import *  # noqa: E402,F401,F403
from polypack.scrape_tables import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
