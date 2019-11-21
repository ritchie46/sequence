import numpy as np
import torch


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
        self.data = np.array(paths)
        self.language = language
        self.transform_data()
        # used for shuffling
        self.idx = np.arange(len(self.data))

    def transform_sentence(self, s):
        idx = []
        for w in s:
            if w in self.skip:
                continue
            if w not in self.language.w2i:
                self.language.register_single_word(w)
            idx.append(self.language.w2i[w])
        idx.append(0)
        return torch.tensor(idx)

    def transform_data(self):
        self.data = list(
            filter(lambda x: len(x) > 0, map(self.transform_sentence, self.data))
        )

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
        x = [self.data[i].to(device) for i in idx]
        padded = torch.nn.utils.rnn.pad_sequence(x, padding_value=-1)

        return (
            torch.nn.utils.rnn.pack_padded_sequence(
                padded, lengths=[len(t) for t in x], enforce_sorted=False
            ),
            padded,
        )

    def __len__(self):
        return len(self.data)
