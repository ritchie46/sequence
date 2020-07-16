import numpy as np
import torch
from collections import defaultdict
import string
from sequence.data import traits
from enum import IntEnum
from typing import List, Dict, Optional, Union, Sequence
import functools


def lazyprop(fn):
    attr_name = "_lazy_" + fn.__name__

    @property
    @functools.wraps(fn)
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


class Tokens(IntEnum):
    EOS = 0
    SOS = 1
    UNKNOWN = 2


class Language:
    """
    The vocabulary of a dataset. This will map the unique strings to indexes that map to the embeddings.
    You can also provide custom embeddings object.
    """

    def __init__(
        self,
        words: Optional[List[str]] = None,
        lower: bool = True,
        remove_punctuation: bool = True,
        custom_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        words
            Unique words in the dataset
        lower
            Lower the string: str.lower()
        remove_punctuation
            !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ will be removed
        custom_embeddings
            Shape(num_embeddings, embedding_dim)
        """
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        if remove_punctuation:
            self.translation_table = str.maketrans("", "", string.punctuation)
        else:
            self.remove_punctuation = None
        # Warning. Don't change index 0, 1, and 2
        # These are used in the models!
        self.w2i = {"EOS": Tokens.EOS, "SOS": Tokens.SOS, "UNKNOWN": Tokens.UNKNOWN}
        if words is not None:
            self.register(words)
        self.custom_embeddings = self.init_custom_emb(custom_embeddings)

    @staticmethod
    def init_custom_emb(
        custom_embeddings: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if custom_embeddings is not None:
            # we need to concat 3 embeddings for EOS, SOS and UNKNOWN
            row, emb_dim = custom_embeddings.shape

            pre = np.zeros((row, emb_dim))
            # one hot encode. TODO: maybe something smarter?
            # can fail if embedding size is two. It happened to you? You can't be serious!
            pre[0, 0] = 1
            pre[1, 1] = 1
            pre[2, 2] = 1
            return np.concatenate((pre, custom_embeddings), axis=0)

    def clean(self, word: str) -> str:
        """
        Transform str to lowercase and optionally remove punctuation
        """
        if self.lower:
            word = word.lower()
        if self.remove_punctuation:
            # Make a translation table when given 3 args.
            # All punctuation will be mapped to None
            word = word.translate(self.translation_table)
        return word

    def register(self, words: List[str]):
        [self.register_single_word(w) for w in words]

    def register_single_word(self, word: str):
        """
        Register words as indexes

        Parameters
        ----------
        word
            unique word
        """
        c = self.clean(word)
        if len(c) > 0:
            self.w2i[c] = len(self.w2i)

    @lazyprop
    def i2w(self) -> Dict[int, Optional[str]]:
        d = defaultdict(lambda: None)
        d.update({v: k for k, v in self.w2i.items()})
        return d

    @property
    def vocabulary_size(self) -> int:
        return len(self.w2i)

    @property
    def words(self) -> List[str]:
        return list(self.w2i.keys())

    def translate_batch(self, padded: torch.tensor) -> np.ndarray:
        """

        Parameters
        ----------
        padded : torch.Tensor
            Tensor with word indexes. Shape: (seq_len, batch)

        Returns
        -------
        out : np.Array[int]
            Array with matching words. Shape: (seq_len, batch)
        """
        # Only eval once
        d = self.i2w
        d[-1] = ""

        if hasattr(padded, "cpu"):
            padded = padded.cpu().data.numpy()
        return np.vectorize(d.get)(padded)

    def __getitem__(self, item: Union[int, str]) -> Union[int, str]:
        if isinstance(item, int):
            return self.i2w[item]
        else:
            return self.w2i[item]

    def __contains__(self, item: Union[int, str]) -> bool:
        if isinstance(item, str):
            return self.clean(item) in self.w2i
        else:
            return item in self.i2w


class Dataset(
    traits.Query, traits.TransitionMatrix, traits.Transform, traits.DatasetABC
):
    """
    Dataset used in training. This has some lazy operations due to dask usage.
    """

    def __init__(
        self,
        sentences: List[List[str]],
        language: Language,
        skip: Sequence[str] = (),
        buffer_size: int = int(1e4),
        max_len: Optional[int] = None,
        min_len: int = 1,
        device: str = "cpu",
        chunk_size: Union[int, str] = "auto",
        allow_con_dup: bool = True,
    ):
        """
        Parameters
        ----------
        sentences : list[list[str]]
            [["hello", "world!"], ["get", "down!"]]
        language : sequence.data.utils.Language
            Required. Should be the language fitted for training.
        skip : list[str]
            Words to skip.
        buffer_size : int
            Size of chunks prepared by lazy generator.
            Only used during preparation of dataset.
        max_len : int
            Max sequence length.
        min_len : int
            Min sequence length.
        device : str
            'cuda' | 'cpu'
        chunk_size : str/ int
            Passed to dask array.
        allow_con_dup : bool
            Filter sequences from consecutive duplicates
        """

        language = Language() if language is None else language
        traits.DatasetABC.__init__(self, self, language=language, device=device)
        traits.Query.__init__(self, self)
        traits.TransitionMatrix.__init__(self, self)
        traits.Transform.__init__(
            self,
            parent=self,
            buffer_size=buffer_size,
            min_len=min_len,
            max_len=max_len,
            chunk_size=chunk_size,
            sentences=sentences,
            skip=skip,
            allow_con_dup=allow_con_dup,
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


class ArrayWrap(np.ndarray):
    # We only wrap a numpy array such that it has a compute method
    # See: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    def __new__(cls, input_array, attr=None):
        obj = np.asarray(input_array).view(cls)
        obj.compute = attr
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.compute = lambda: self


class DatasetEager(Dataset):
    """
    The eager variation of Dataset. This class doesn't use Dask.
    """

    def __init__(
        self,
        sentences: List[List[str]],
        language: Optional[Language] = None,
        skip: Sequence[str] = (),
        buffer_size: int = int(1e4),
        max_len: Optional[int] = None,
        min_len: int = 1,
        device: str = "cpu",
        chunk_size: str = "auto",
    ):
        super().__init__(
            sentences=sentences,
            language=language,
            skip=skip,
            buffer_size=buffer_size,
            max_len=max_len,
            min_len=min_len,
            device=device,
            chunk_size=chunk_size,
        )

    def transform_data(self):
        if self.max_len is None:
            self.max_len = max(map(len, self.paths))

        size = len(self.paths)
        array = []
        for i, j in zip(
            range(0, size, self.buffer_size),
            range(self.buffer_size, size + self.buffer_size, self.buffer_size),
        ):
            array.append(self._gen(i, j, size))

        self.data = np.concatenate(array)

        # Because of transformation conditions there can be empty sequences
        # These need to be removed.

        # The actual computed values are a bit shorter.
        # Because the data rows % buffer_size has a remainder.
        mask_short = self.data.sum(-1) == -(self.max_len + 1)
        mask = np.ones(shape=(self.data.shape[0],), dtype=bool)
        mask[: mask_short.shape[0]] = mask_short

        self.data = ArrayWrap(self.data[~mask])
        self.set_idx()


class DatasetInference(traits.Query, traits.Transform, traits.DatasetABC):
    """
    Dataset used during inference.
    """

    def __init__(
        self,
        sentences: List[List[str]],
        language: Language,
        buffer_size: int = int(1e4),
        max_len: Optional[int] = None,
        min_len: int = 1,
        device: str = "cpu",
        chunk_size: str = "auto",
    ):
        traits.DatasetABC.__init__(self, self, language=language, device=device)
        traits.Query.__init__(self, self)
        traits.Transform.__init__(
            self,
            parent=self,
            buffer_size=buffer_size,
            min_len=min_len,
            max_len=max_len,
            chunk_size=chunk_size,
            sentences=sentences,
            skip=(),
            allow_con_dup=False,
            mask=False,
        )

    def transform_sentence(self, s: List[str]) -> np.ndarray:
        """
        Transform sentence of string to integers.

        This method is different from the one in training because we
        don't want to add new words to the language. Unknown words will
        be added to UNKNOWN.

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
        last_w = None

        if len(s) > self.max_len or len(s) < self.min_len:
            # will be removed jit
            return idx

        i = -1
        for w in s:
            if not self.allow_duplicates:
                if w == last_w:
                    last_w = w
                    continue
                last_w = w
            # Only increment if we don't continue
            i += 1

            if w not in self.language.w2i:
                w = Tokens.UNKNOWN.name
            idx[i] = self.language.w2i[w]
        idx[i + 1] = 0
        return np.array(idx)
