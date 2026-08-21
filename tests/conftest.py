import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SOLUTIONS = REPO / "paper" / "solutions"


@pytest.fixture(scope="session")
def solutions_dir():
    """The 55 published claim coordinate sets (paper/solutions/) double as the
    test corpus: every value asserted here is also a value claimed publicly."""
    return SOLUTIONS
