import torch
import numpy as np
from sequence.model.vae import VAE, det_neg_elbo, run_decoder
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
    det_neg_elbo(m, packed_padded, 1., test_loss=True)

