import torch
from sequence.test import language, words, dataset, paths
from sequence.model.stamp import (
    STMP,
    trilinear_composition,
    det_loss,
    STAMP,
    AttentionNet,
)
import numpy as np


def test_flow(language, dataset):
    torch.manual_seed(0)
    m = STMP(language.vocabulary_size, embedding_dim=2)
    batch_size = 3

    with torch.no_grad():
        packed_padded, padded = dataset.get_batch(0, batch_size)
        m_s, m_t, h_s, h_t, x, z, y_hat = m(packed_padded, return_all=True)

        assert m_s.shape == m_t.shape
        assert np.allclose(m_s[0, 0, :], m_t[0, 0, :])
        # Check single batch sequence
        # l,b,e
        m_t = m_t[:, 0, :]
        m_s = m_s[:, 0, :]
        assert np.allclose(
            np.cumsum(m_t, 0) / torch.arange(1, len(m_t) + 1).reshape(-1, 1), m_s
        )


def test_loss(language, dataset):
    torch.manual_seed(0)
    m = STMP(language.vocabulary_size, embedding_dim=2)
    batch_size = 3

    with torch.no_grad():
        packed_padded, padded = dataset.get_batch(0, batch_size)
        det_loss(m, packed_padded)


def test_trilinear_composition():
    torch.manual_seed(0)
    b = 3
    e = 2
    v = 11
    l = 6

    h_s = torch.rand(l, b, e)
    h_t = torch.rand(l, b, e)
    x = torch.rand(v, e)

    a = trilinear_composition(h_s, h_t, x, einsum=True)

    b = trilinear_composition(h_s, h_t, x, einsum=False)

    np.testing.assert_almost_equal(a.numpy(), b.numpy())

    # Manual check of trilinear composition

    # l = 2
    # b = 2
    # v = 2
    # e = 2

    # Sequences as 2D embedding vectors.

    # seq1: 1, 2, 3
    #       1, 2, 3

    # seq2: 4, 5, 6
    #       4, 5, 6

    h_t = torch.tensor(
        [[[1, 1], [4, 4]], [[2, 2], [5, 5]], [[3, 3], [6, 6]]], dtype=float
    )

    # h_s = cumulative average

    # cumsum
    # -------
    # seq1: 1, 3, 6

    # seq2: 4, 9, 15

    # cum average
    # -----------
    # seq1: 1, 1.5, 2

    # seq2: 4, 4.5, 5

    cumsum = torch.cumsum(h_t, 0)
    lengths = torch.arange(1, 1 + cumsum.shape[0])
    h_s = cumsum / lengths.reshape(cumsum.shape[0], 1, 1)

    assert np.all(np.isclose(h_s[:, 0, 0], np.array([1.0, 1.5, 2.0])))

    # vocabulary can go to 6
    x = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=float)

    # b,l,v
    a = trilinear_composition(h_s, h_t, x)

    # batch 1, length 1, item 1
    # <h_s, h_t, x_i>
    h_s = np.array([1, 1])
    h_t = np.array([1, 1])
    x_i = np.array([1, 1])
    assert np.isclose(np.dot(h_s, (h_t * x_i)), a[0, 0, 0])

    # batch 2, length 3, item 3
    h_s = np.array([5, 5])
    h_t = np.array([6, 6])
    x_i = np.array([3, 3])
    assert np.isclose(np.dot(h_s, (h_t * x_i)), a[1, 2, 2])


def test_flow_stamp(language, dataset):
    torch.manual_seed(0)
    m = STAMP(language.vocabulary_size, embedding_dim=2)
    batch_size = 3

    with torch.no_grad():
        packed_padded, padded = dataset.get_batch(0, batch_size)
        m(packed_padded)

    det_loss(m, packed_padded, test_loss=True)


def test_attention():
    torch.manual_seed(0)
    # Attention head for embedding size of 2
    att = AttentionNet(2)

    # Two sessions s1, and s2
    s1 = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.float)

    s2 = torch.tensor([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])

    m_s = s1.sum(0)

    # Paper STAMP eq. 7
    # compute a_0
    x_0 = s1[0, :]

    x_t = s1[-1, :]
    a_0 = att.w0(torch.sigmoid(att.w1(x_0) + att.w2(x_t) + att.w3(m_s)))

    # Now for a batch
    # l, b, e
    e = torch.stack([s1, s2], dim=1)

    _, ai = att(e, return_attention_factors=True)
    assert ai[-1][0, 0] == a_0
