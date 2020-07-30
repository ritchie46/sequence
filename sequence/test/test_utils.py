from torch.utils.data import DataLoader
from sequence.data.datasets import brown
from sequence.data.utils import DatasetInference, Tokens, Language
from sequence.test import language, words, dataset, paths
import pytest
import numpy as np
import random


def test_dataset_torch_compatible(dataset):
    dl = DataLoader(dataset)
    # assert runs
    next(iter(dl))
    dl = DataLoader(dataset, shuffle=True)
    next(iter(dl))


def test_brown_dataset():
    ds, lang = brown()
    assert lang[234] == "found"
    # Check if punctuation is removed.
    with pytest.raises(ValueError):
        lang.words.index(".")
    ds.get_batch(0, 10)


def test_dataset_split(dataset):
    dataset.max_len = 10
    dataset.min_len = 5
    ds_train, ds_test = dataset.split([0.8, 0.2])
    assert ds_train.min_len == dataset.min_len
    assert ds_test.max_len == dataset.max_len
    assert ds_test.data.shape == (400, 9)
    assert ds_train.data.shape == (1600, 9)


def test_transition_matrix(dataset):
    mm = dataset.transition_matrix
    assert mm[0, 1] == 0


def test_inference_dset(paths, dataset):
    language = dataset.language

    # Shuffle paths so that the dataset cannot create the same data.
    np.random.seed(1)
    np.random.shuffle(paths)

    new_paths = []
    for p in paths:
        new = list(p)
        new.append(random.choice("ZYZWVUT"))
        new_paths.append(new)

    inference_ds = DatasetInference(sentences=new_paths, language=language)
    # Assert that the new words are assigned to the UNKNOWN field.
    assert (inference_ds.data.compute() == Tokens.UNKNOWN.value).sum() > 0
    assert (dataset.data.compute() == Tokens.UNKNOWN.value).sum() == 0
    print(inference_ds.data)
    assert len(inference_ds) == len(new_paths)


def test_custom_emb():
    emb = np.random.rand(3, 5)
    lang = Language(custom_embeddings=emb)
    # check if eos, sos and unknown are inserted
    assert lang.custom_embeddings.shape == (6, 5)
