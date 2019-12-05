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
