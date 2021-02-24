import pytest
from pathlib import Path

work_dir = Path(__file__).parent 

@pytest.fixture(scope='module', params=['mypkg', 'fail'])
def pkg(request):
    return request.param

@pytest.fixture
def data(pkg):
    return 1 if pkg == 'mypkg' else 0

@pytest.fixture
def sample_data():
    return 1

def test_01(sample_data):
    assert sample_data

def test_02(data):
    assert data

@pytest.mark.parametrize("param", [0, 1])
def test_03(param):
    assert bool(param)

@pytest.mark.parametrize(["param", "expected"], 
                         [(0, False),
                          (1, True)]
                         )
def test_04(param, expected):
    assert bool(param) == expected
