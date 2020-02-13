import torch
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import logging
from sequence.utils.general import TqdmLoggingHandler
import tqdm
import dask.array as da
import dask

logger = logging.getLogger(__name__)
logger.addHandler(TqdmLoggingHandler())


class DatasetABC:
    def __init__(self, parent, language, device):
        self.parent = parent
        self.language = language
        self.device = device
        self.data = np.array([[]])


class Query:
    """
    Required: DatasetABC
    """
    def __init__(self, parent):
        self.parent = parent
        # used for shuffling
        self.idx = None

    def set_idx(self):
        self.idx = np.arange(len(self.parent.data), dtype=np.int32)

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
        idx = self.parent.idx[start:end]
        return self.__getitem__(idx, device)

    def __len__(self):
        return len(self.parent.data)

    def __getitem__(self, idx, device=None):
        if isinstance(idx, int):
            idx = [idx]
        x = self.parent.data[idx].compute()

        if device is None:
            device = self.parent.device

        idx_cond = np.argwhere(x == 0)
        lengths = idx_cond[np.unique(idx_cond[:, 0], return_index=True)[1]][:, 1] + 1

        padded = torch.tensor(x.T, dtype=torch.long).to(device)[: max(lengths) + 1, :]
        return (
            torch.nn.utils.rnn.pack_padded_sequence(
                padded, lengths=lengths, enforce_sorted=False
            ),
            padded,
        )

    def get_single_row(self, i):
        # remove the EOS and -1 Pad
        return self.__getitem__(i)[1].T.flatten()[:-2]

    def shuffle(self):
        np.random.shuffle(self.parent.idx)


class TransitionMatrix:
    """
    Required: Query, DatasetABC
    """

    def __init__(self, parent):
        self.parent = parent
        self._trans_matrix = None

    @property
    def transition_matrix(self):
        if self._trans_matrix is None:
            logger.info("creating transition matrix...")
            rank = self.parent.language.vocabulary_size
            mm = lil_matrix((rank, rank), dtype=np.float32)
            for k in tqdm.tqdm(range(self.parent.data.shape[0])):
                row = self.parent.get_single_row(k)

                for i, j in zip(row, row[1:]):
                    mm[i, j] += 1.0

            # reshape such that w/ broadcasting we divide the whole rows
            total = mm.sum(0).reshape(-1, 1)
            # but first, we cannot divide by 0
            total[total == 0] = 1.0
            mm = mm / total
            self._trans_matrix = csr_matrix(mm)
        return self._trans_matrix


class Transform:
    """
    Required: DatasetABC
    """
    def __init__(
        self,
        parent,
        buffer_size,
        min_len,
        max_len,
        chunk_size,
        sentences,
        skip,
        allow_con_dup,
    ):
        self.parent = parent
        self.buffer_size = buffer_size
        self.min_len = min_len
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.skip = set(skip)
        self.allow_duplicates = allow_con_dup

        if sentences is not None:
            self.paths = sentences
            self.transform_data()
            self.paths = None  # Free memory

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
        s = list(
            filter(lambda x: len(x) > 0, [self.parent.language.clean(w) for w in s])
        )

        # All the sentences are -1 padded
        idx = np.ones(self.max_len + 1) * -1
        last_w = None

        if len(s) > self.max_len or len(s) < self.min_len:
            # will be removed jit
            return idx

        i = -1
        for w in s:
            if w in self.skip:
                continue
            if not self.allow_duplicates:
                if w == last_w:
                    last_w = w
                    continue
                last_w = w
            # Only increment if we don't continue
            i += 1

            if w not in self.parent.language.w2i:
                self.parent.language.register_single_word(w)
            idx[i] = self.parent.language.w2i[w]
        idx[i + 1] = 0
        return np.array(idx)

    # https://blog.dask.org/2019/06/20/load-image-data
    def _gen(self, i, j, size):
        """
        Note: Function has one time side effect due
        to self.transform_sentence
        """
        j = min(size, j)

        # Sentences to integers
        a = np.array(
            list(map(self.transform_sentence, self.paths[i:j])), dtype=np.int32
        )
        return a

    def transform_data(self):
        """
        The sentences containing of string values will be
        transformed to a dask dataframe as integers.

        Parameters
        ----------
        max_len : int
            Maximum length to use in the dataset.
        """
        if self.max_len is None:
            self.max_len = max(map(len, self.paths))

        size = len(self.paths)
        lazy_a = []

        for i, j in zip(
            range(0, size, self.buffer_size),
            range(self.buffer_size, size + self.buffer_size, self.buffer_size),
        ):
            lazy_a.append(dask.delayed(self._gen)(i, j, size))

        lazy_a = [
            da.from_delayed(
                x, shape=(self.buffer_size, self.max_len + 1,), dtype=np.int32
            )
            for x in lazy_a
        ]
        self.parent.data = da.concatenate(lazy_a)

        # Because of transformation conditions there can be empty sequences
        # These need to be removed.

        # The actual computed values are a bit shorter.
        # Because the data rows % buffer_size has a remainder.

        mask_short = self.parent.data.sum(-1).compute() == -(self.max_len + 1)
        mask = np.ones(shape=(self.parent.data.shape[0],), dtype=bool)
        mask[: mask_short.shape[0]] = mask_short

        self.parent.data = self.parent.data[~mask]
        self.parent.data = self.parent.data.persist()
        self.parent.set_idx()
