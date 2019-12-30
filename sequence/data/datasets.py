import nltk
from sequence.data.utils import Dataset


def brown():
    nltk.download("brown")
    ds = Dataset(nltk.corpus.brown.sents())
    return ds, ds.language


def treebank():
    nltk.download("treebank")
    ds = Dataset(nltk.corpus.treebank.sents())
    return ds, ds.language
