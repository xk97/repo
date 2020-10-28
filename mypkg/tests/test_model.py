import pytest

@pytest.fixture
def fix():
    return 1

def test_01(fix):
    """
    docstring
    """
    pass