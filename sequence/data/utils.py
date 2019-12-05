import numpy as np
import torch
import dask.array as da
from torch.utils.data import Dataset as ds


class Language:
    def __init__(self, words=None):
        self.w2i = {"EOS": 0, "SOS": 1, "UNKNOWN": 2}
        if words is not None:
            self.register(words)

    def register(self, words):
        for i in range(len(words)):
            self.w2i[words[i]] = i + 3

    def register_single_word(self, word):
        self.w2i[word] = len(self.w2i)

    @property
    def i2w(self):
        return {v: k for k, v in self.w2i.items()}

    @property
    def vocabulary_size(self):
        return len(self.w2i)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2w[item]
        else:
            return self.w2i[item]


class Dataset(ds):
    def __init__(
        self,
        sentences,
        language=None,
        skip=(),
        chunk_size=int(1e4),
        max_len=None,
        min_len=1,
        device="cpu",
    ):
        self.skip = set(skip)
        self.data = np.array([[]])
        self.max_len = max_len
        self.min_len = min_len
        self.chunk_size = chunk_size
        # used for shuffling
        self.idx = None
        self.language = Language() if language is None else language
        self.transform_data(sentences)
        self.device = device

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

    def transform_data(self, paths, max_len=None):
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
        if max_len is None:
            max_len = max(map(len, paths))
        self.max_len = max_len

        size = len(paths)
        for i, j in zip(
            range(0, size, self.chunk_size),
            range(self.chunk_size, size + self.chunk_size, self.chunk_size),
        ):
            j = min(size, j)

            # Sentences to integers
            a = np.array(list(map(self.transform_sentence, paths[i:j])))

            # Remove empty sequences jit. Because of -1 padding sum == max_len + 1
            mask = np.sum(a, 1) == -(self.max_len + 1)
            a = a[~mask]

            self.data = da.concatenate([self.data, a], axis=0)

        self.idx = np.arange(len(self.data), dtype=np.int32)

    def shuffle(self):
        np.random.shuffle(self.idx)
        # used for shuffling
        self.idx = np.arange(len(self.data))

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
