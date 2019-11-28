from itertools import permutations
from clickstream.data.utils import Language, Dataset
from clickstream.seq2seq import EncoderDecoder, decoder_loss
import pytest
import random
import torch
import numpy as np
from clickstream import utils


@pytest.fixture(scope="module")
def words():
    return list("ABCDEFGH")


@pytest.fixture(scope="module")
def paths(words):
    # Create random sequences with random length
    random.seed(1)
    perm = permutations("".join(words))
    return [next(perm)[: random.choice(range(1, 9))] for _ in range(2000)]


@pytest.fixture(scope="module")
def language(words):
    return Language(words)


@pytest.fixture(scope="module")
def dataset(paths, language):
    return Dataset(paths, language)


def test_encoder_decoder_flow(dataset, language):
    latent_size = 20
    batch_size = 3
    m = EncoderDecoder(
        vocabulary_size=language.vocabulary_size,
        embedding_dim=10,
        latent_size=latent_size,
        bidirectional=False,
    )
    packed_padded, padded = dataset.get_batch(0, batch_size)
    z = m.encode(packed_padded)

    # First word should be initiated in the encode method
    out, h = m.decode(word=None, h=z)
    assert out.shape == (batch_size, language.vocabulary_size)
    # assert h.shape[1:] == (batch_size, latent_size)

    # Loss should be non zero
    loss = decoder_loss(m, padded)
    assert loss > 0

    # Only the last row has a v
    z_ = m.encode(
        torch.nn.utils.rnn.pack_padded_sequence(
            torch.cat([padded, torch.tensor([[-1, -1, -1]])], dim=0),
            enforce_sorted=False,
            lengths=[4, 3, 6],
        )
    )
    if isinstance(z, tuple):
        z = z[0]
        z_ = z_[0]

    assert torch.all(z == z_)
    assert z.shape[1:] == (batch_size, latent_size)


def test_reverse_target():
    sequences = [[1, 2, 3], [1, 2, 3, 4, 5], [1]]
    sequences = [torch.tensor(a) for a in sequences]
    lengths = [3, 5, 1]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, padding_value=-1)
    padded_new = utils.masked_flip(padded.T, lengths)
    np.testing.assert_allclose(
        np.array([[3, 5, 1], [2, 4, -1], [1, 3, -1], [-1, 2, -1], [-1, 1, -1]]),
        padded_new.numpy(),
    )


def test_dask_arrays(paths, language):
    ds = Dataset(paths, language)
    packed_padded, padded = ds.get_batch(58, 64)
    assert torch.all(packed_padded.data != -1)
    assert ds.data.shape == (2000, 9)

    padded_, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_padded, padding_value=-1)
    np.testing.assert_allclose(padded, padded_)


def test_custom_embedding_initialization():
    vocab_size = 10
    embedding_size = 3
    w2v = torch.ones(vocab_size, embedding_size)
    m = EncoderDecoder(
        vocabulary_size=vocab_size, embedding_dim=None, custom_embeddings=w2v
    )
    emb = m.emb(torch.tensor([1], dtype=torch.long))
    np.testing.assert_allclose(emb.numpy(), np.ones((1, 3)))


def test_rnn_init():
    m = EncoderDecoder(vocabulary_size=1, rnn_type="lstm")
    assert isinstance(m.rnn_enc, torch.nn.LSTM)
    m = EncoderDecoder(vocabulary_size=1, rnn_type="gru")
    assert isinstance(m.rnn_enc, torch.nn.GRU)
