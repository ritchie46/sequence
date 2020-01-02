import torch
import numpy as np
from sequence.model.vae import VAE, det_neg_elbo, run_decoder, inference
from sequence.test import language, words, dataset, paths


def test_flow(language, dataset):
    m = VAE(language.vocabulary_size)
    batch_size = 3
    with torch.no_grad():
        packed_padded, padded = dataset.get_batch(0, batch_size)
        h, z, mu, log_var = m.encode(packed_padded)
        out, target = run_decoder(m, packed_padded, word_dropout=1, h=h)
        assert np.all(np.isclose(out.exp().sum(-1), 1))


def test_loss(language, dataset):
    m = VAE(language.vocabulary_size)
    batch_size = 3
    packed_padded, padded = dataset.get_batch(0, batch_size)
    det_neg_elbo(m, packed_padded, 1.0, test_loss=True)


def test_inference(language, dataset):
    m = VAE(language.vocabulary_size, bidirectional=False)
    batch_size = 3
    m.eval()

    with torch.no_grad():
        packed_padded, padded = dataset.get_batch(0, batch_size)
        out1 = inference(m, packed_padded, n=10)

    with torch.no_grad():
        out2 = inference(m, packed_padded, n=20)

    # The first part of both predictions should be equal. As it is a unidirectional RNN.
    np.testing.assert_allclose(out1, out2[:, :10])
