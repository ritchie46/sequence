import nltk
from sequence.data.utils import Dataset
from urllib import request
import os
import logging
from pyunpack import Archive, PatoolError
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def brown():
    nltk.download("brown")
    ds = Dataset(nltk.corpus.brown.sents())
    return ds, ds.language


def treebank():
    nltk.download("treebank")
    ds = Dataset(nltk.corpus.treebank.sents())
    return ds, ds.language


def download_and_unpack_yoochoose(storage_dir):
    fp = os.path.join(storage_dir, "yoochoose-data.7z")
    if not os.path.isfile(fp):
        logger.info("Downloading Yoochoose dataset...")
        url = "https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z"
        request.urlretrieve(url, fp)
    else:
        logger.info((f"Yoochoose dataset already exists in {fp}"))

    ds = os.path.join(storage_dir, "yoochoose-data")
    if not os.path.isdir(ds):
        logger.info((f"Unpacking zip archive"))
        os.makedirs(ds)

        try:
            Archive(fp).extractall(ds)
        except PatoolError as e:
            logger.error(
                f"{e}\nInstall a system application to process 7zip files, such as p7zip."
            )
            os.removedirs(ds)


def yoochoose(dn, nrows=None, min_unique=5, skiprows=None, div64=False, test=False):
    """

    Parameters
    ----------
    dn : str
        Directory path
    nrows : Union[None, int]
        Take only n_rows from the dataset.
    min_unique : int
        Items that occur less than min_unique are removed.
    skiprows : Union[None, int]
        Skip rows from csv.
    div64 : bool
        Load yoochoose 1/64
    test : bool
        Load test set

    Returns
    -------
    (ds, lang) : tuple[
                    sequence.data.utils.Dataset,
                    sequence.data.utils.Language
                    ]
    """
    if test:
        fn = "yoochoose-data/yoochoose-test.dat"
    else:
        fn = "yoochoose-data/yoochoose-clicks.dat"
    df = pd.read_csv(
        os.path.join(dn, fn),
        names=["session_id", "timestamp", "item_id", "category"],
        usecols=["session_id", "item_id"],
        dtype={"session_id": np.int32, "item_id": np.str},
        nrows=nrows,
        skiprows=skiprows
    )

    # Series of item_id -> counts
    item_n_unique = df["item_id"].value_counts()
    # Filter counts < k
    item_n_unique = item_n_unique[item_n_unique > min_unique]

    # Create df[item_id, counts]
    item_n_unique = (
        item_n_unique.to_frame("counts")
        .reset_index()
        .rename(columns={"index": "item_id"})
    )
    df = df.merge(item_n_unique, how="inner", on="item_id")[["session_id", "item_id"]]

    if div64:
        n = df.shape[0] // 64
        df = df.iloc[-n:]

    agg = df.groupby("session_id").agg(list)
    ds = Dataset([r[1] for r in agg.itertuples()])
    return ds, ds.language


