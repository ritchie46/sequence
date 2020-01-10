import pytest
from sequence.data import datasets
import os


@pytest.fixture()
def yoochoose_dir():
    storage_dir = 'deleteme'
    os.makedirs(storage_dir, exist_ok=True)
    datasets.download_and_unpack_yoochoose(storage_dir)

    yield storage_dir


def test_yoochoose(yoochoose_dir):
    ds, lang = datasets.yoochoose(yoochoose_dir, nrows=5000)
