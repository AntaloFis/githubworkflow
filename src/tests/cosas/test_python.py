import pytest


@pytest.mark.usefixtures("list_data")
def test_all(list_data):
    print(list_data)

    assert isinstance(list_data, list)