import nltk
from sequence.data.utils import Dataset


def brown():
    nltk.download("brown")
    ds = Dataset(nltk.corpus.brown.sents())
    return ds, ds.language
