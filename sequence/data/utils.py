import numpy as np
import dask
import torch
import dask.array as da
from torch.utils.data import Dataset as ds
from collections import defaultdict
import string


class Language:
    def __init__(self, words=None, lower=True, remove_punctuation=True):
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        if remove_punctuation:
            self.translation_table = str.maketrans("", "", string.punctuation)
        else:
            self.remove_punctuation = None
        # Warning. Don't change index 0, 1, and 2
        # These are used in the models!
        self.w2i = {"EOS": 0, "SOS": 1, "UNKNOWN": 2}
        if words is not None:
            self.register(words)

    def clean(self, word):
        if self.lower:
            word = word.lower()
        if self.remove_punctuation:
            # Make a translation table when given 3 args.
            # All punctuation will be mapped to None
            word = word.translate(self.translation_table)
        return word

    def register(self, words):
        [self.register_single_word(w) for w in words]

    def register_single_word(self, word):
        c = self.clean(word)
        if len(c) > 0:
            self.w2i[c] = len(self.w2i)

    @property
    def i2w(self):
        d = defaultdict(lambda: None)
        d.update({v: k for k, v in self.w2i.items()})
        return d

    @property
    def vocabulary_size(self):
        return len(self.w2i)

    @property
    def words(self):
        return list(self.w2i.keys())

    def translate_batch(self, padded):
        """

        Parameters
        ----------
        padded : torch.Tensor
            Tensor with word indexes. Shape: (seq_len, batch)

        Returns
        -------
        out : np.Array
            Array with matching words. Shape: (seq_len, batch)
        """
        # Only eval once
        d = self.i2w
        d[-1] = ""

        if hasattr(padded, "cpu"):
            padded = padded.cpu().data.numpy()
        return np.vectorize(d.get)(padded)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2w[item]
        else:
            return self.w2i[item]

    def __contains__(self, item):
        if isinstance(item, str):
            return self.clean(item) in self.w2i
        else:
            return item in self.i2w


class Dataset(ds):
    def __init__(
        self,
        sentences,
        language=None,
        skip=(),
        buffer_size=int(1e4),
        max_len=None,
        min_len=1,
        device="cpu",
        chunk_size="auto",
    ):
        self.skip = set(skip)
        self.data = np.array([[]])
        self.max_len = max_len
        self.min_len = min_len
        self.buffer_size = buffer_size
        # used for shuffling
        self.idx = None
        self.language = Language() if language is None else language
        if sentences is not None:
            self.transform_data(sentences)
        self.device = device
        self.chunk_size = chunk_size

    def transform_sentence(self, s):
        """
        Transform sentence of string to integers.

        Parameters
        ----------
        s : list[str]
            A sequence of any length.

        Returns
        -------
        s : np.array[int]
            A -1 padded sequence of shape (self.max_len, )
        """
        s = list(filter(lambda x: len(x) > 0, [self.language.clean(w) for w in s]))

        # All the sentences are -1 padded
        idx = np.ones(self.max_len + 1) * -1

        if len(s) > self.max_len or len(s) < self.min_len:
            # will be removed jit
            return idx

        for i, w in enumerate(s):
            if w in self.skip:
                continue

            if w not in self.language.w2i:
                self.language.register_single_word(w)
            idx[i] = self.language.w2i[w]
        idx[i + 1] = 0
        return np.array(idx)

    def transform_data(self, paths):
        """
        The sentences containing of string values will be
        transformed to a dask dataframe as integers.

        Parameters
        ----------
        paths : list[list[str]]
            Sentences with a variable length.
        max_len : int
            Maximum length to use in the dataset.
        """
        if self.max_len is None:
            self.max_len = max(map(len, paths))

        size = len(paths)

        # https://blog.dask.org/2019/06/20/load-image-data
        def gen(i, j):
            """
            Note: Function has one time side effect due
            to self.transform_sentence
            """
            j = min(size, j)

            # Sentences to integers
            a = np.array(list(map(self.transform_sentence, paths[i:j])), dtype=np.int32)
            return a

        lazy_a = []
        for i, j in zip(
                range(0, size, self.buffer_size),
                range(self.buffer_size, size + self.buffer_size, self.buffer_size),
        ):
            lazy_a.append(dask.delayed(gen)(i, j))

        lazy_a = [da.from_delayed(x, shape=(self.buffer_size, self.max_len + 1,), dtype=np.int32) for x in lazy_a]

        self.data = da.concatenate(lazy_a)

        # Because of transformation conditions there can be empty sequences
        # These need to be removed.

        # The actual computed values are a bit shorter.
        # Because the data rows % buffer_size has a remainder.
        mask_short = self.data.sum(-1).compute() == -(self.max_len + 1)
        mask = np.ones(shape=(self.data.shape[0],), dtype=bool)
        mask[:mask_short.shape[0]] = mask_short

        self.data = self.data[~mask]
        self.data = self.data.persist()
        self.set_idx()

    def shuffle(self):
        np.random.shuffle(self.idx)

    def get_batch(self, start, end, device="cpu"):
        """
        Get a slice from the dataset.

        Parameters
        ----------
        start : int
        end : int
        device : str
            'cpu' or 'cuda'

        Returns
        -------
        out : [PackedPaddedSequence, PaddedSequence]
            shape: (seq_len, batch)

        """
        idx = self.idx[start:end]
        return self.__getitem__(idx, device)

    def set_idx(self):
        self.idx = np.arange(len(self.data), dtype=np.int32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, device=None):
        if isinstance(idx, int):
            idx = [idx]
        x = self.data[idx].compute()

        if device is None:
            device = self.device

        idx_cond = np.argwhere(x == 0)
        lengths = idx_cond[np.unique(idx_cond[:, 0], return_index=True)[1]][:, 1] + 1

        padded = torch.tensor(x.T, dtype=torch.long).to(device)[: max(lengths) + 1, :]
        return (
            torch.nn.utils.rnn.pack_padded_sequence(
                padded, lengths=lengths, enforce_sorted=False
            ),
            padded,
        )

    def split(self, fracs, shuffle=True):
        """
        Split dataset in [train, test, ..., val] Datasets.

        Parameters
        ----------
        fracs : Sequence
        shuffle : bool

        Returns
        -------
        datasets : tuple[Dataset]
            A new Dataset object for every fraction in fracs
        """
        idx = np.arange(len(self.data))
        dsets = tuple(Dataset(None, language=self.language) for _ in fracs)
        fracs = np.array([0] + fracs)
        assert fracs.sum() == 1
        if shuffle:
            np.random.shuffle(idx)

        slice_idx = np.cumsum(fracs * len(self.data)).astype(int)
        slice_idx = [(i, j) for i, j in zip(slice_idx[:-1], slice_idx[1:])]

        for (i, j), ds in zip(slice_idx, dsets):
            ds.__dict__.update(self.__dict__)
            ds.data = self.data[i:j]
            ds.set_idx()
        return dsets
