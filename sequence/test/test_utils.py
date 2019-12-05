from torch.utils.data import DataLoader
from sequence.data.datasets import brown
from sequence.test import language, words, dataset, paths


def test_dataset_torch_compatible(dataset):
    dl = DataLoader(dataset)
    # assert runs
    next(iter(dl))
    dl = DataLoader(dataset, shuffle=True)
    next(iter(dl))


def test_brown_dataset():
    ds, lang = brown()
    assert lang[234] == "burden"


def test_dataset_split(dataset):
    dataset.max_len = 10
    dataset.min_len = 5
    ds_train, ds_test = dataset.split([.8, .2])
    assert ds_train.min_len == dataset.min_len
    assert ds_test.max_len == dataset.max_len
    assert ds_test.data.shape == (400, 9)
    assert ds_train.data.shape == (1600, 9)
