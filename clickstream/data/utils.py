import numpy as np
import torch
import dask.array as da


class Language:
    def __init__(self, words):
        self.w2i = {"EOS": 0, "SOS": 1}
        self.register(words)

    def register(self, words):
        for i in range(len(words)):
            self.w2i[words[i]] = i + 2

    def register_single_word(self, word):
        self.w2i[word] = len(self.w2i)

    @property
    def i2w(self):
        return {v: k for k, v in self.w2i.items()}

    @property
    def vocabulary_size(self):
        return len(self.w2i)


class Dataset:
    def __init__(self, paths, language, skip=()):
        self.skip = set(skip)
        self.data = np.array([])
        self.max_len = None
        # used for shuffling
        self.idx = None
        self.language = language
        self.transform_data(paths)

    def transform_sentence(self, s):
        idx = np.ones(self.max_len + 1) * -1
        for i, w in enumerate(s):
            if w in self.skip:
                continue
            if w not in self.language.w2i:
                self.language.register_single_word(w)
            idx[i] = self.language.w2i[w]
        idx[i + 1] = 0
        return np.array(idx)

    def transform_data(self, paths, max_len=None, chunk_size=1000):
        if max_len is None:
            max_len = max(map(len, paths))
        self.max_len = max_len

        size = len(paths)
        for i, j in zip(
            range(0, size, chunk_size), range(chunk_size, size + chunk_size, chunk_size)
        ):
            j = max(size, j)

            a = np.array(list(map(self.transform_sentence, paths[i:j])))
            mask = np.sum(a, 1) == -(self.max_len + 1)
            a = a[~mask]

        self.data = da.block([self.data, a])
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
        x = self.data[idx].compute()

        idx_cond = np.argwhere(x == 0)
        lengths = idx_cond[np.unique(idx_cond[:, 0], return_index=True)[1]][:, 1] + 1

        padded = torch.tensor(x.T, dtype=torch.long).to(device)[:max(lengths) + 1, :]
        return (
            torch.nn.utils.rnn.pack_padded_sequence(
                padded, lengths=lengths, enforce_sorted=False
            ),
            padded,
        )

    def __len__(self):
        return len(self.data)
