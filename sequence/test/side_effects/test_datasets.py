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


def test_yoochoose(yoochoose_dir):
    agg = datasets.yoochoose(
        yoochoose_dir, nrows=5000, cache=False, return_agg=True, filter_unique=False
    )

    # valid order session 11
    valid_order = [
        "214821275",
        "214821275",
        "214821371",
        "214821371",
        "214821371",
        "214717089",
        "214563337",
        "214706462",
        "214717436",
        "214743335",
        "214826837",
        "214819762",
    ]

    assert list(agg.loc[11])[0] == valid_order


def test_yoochoose_64(yoochoose_dir):
    datasets.yoochoose(
        yoochoose_dir, div64=True, dataset_kwargs=dict(min_len=2, max_len=1000)
    )
