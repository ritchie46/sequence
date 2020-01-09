import torch
from sequence.test import language, words, dataset, paths
from sequence.model.STAMP import STMP, trilinear_composition, det_loss
import numpy as np


def test_flow(language, dataset):
    torch.manual_seed(0)
    m = STMP(language.vocabulary_size, embedding_dim=2)
    batch_size = 3

    with torch.no_grad():
        packed_padded, padded = dataset.get_batch(0, batch_size)
        m(packed_padded)


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
