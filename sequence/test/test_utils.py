from torch.utils.data import DataLoader
from sequence.test import language, words, dataset, paths


def test_(dataset):
    dl = DataLoader(dataset)
    # assert runs
    next(iter(dl))
    dl = DataLoader(dataset, shuffle=True)
    next(iter(dl))
