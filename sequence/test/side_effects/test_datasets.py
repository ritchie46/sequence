import pytest
from sequence.data import datasets
import os
import shutil


@pytest.fixture(scope="session")
def yoochoose_dir():
    storage_dir = "deleteme"
    os.makedirs(storage_dir, exist_ok=True)
    datasets.download_and_unpack_yoochoose(storage_dir)

    yield storage_dir

    shutil.rmtree(storage_dir)


def test_yoochoose_64(yoochoose_dir):
    datasets.yoochoose(
        yoochoose_dir, div64=True, dataset_kwargs=dict(min_len=2, max_len=1000)
    )
