from clickstream.model.vae import VAE, det_elbo
from clickstream.test.test_ae import language, words, dataset, paths


def test_flow(language, dataset):
    m = VAE(language.vocabulary_size)
    batch_size = 3

    packed_padded, padded = dataset.get_batch(0, batch_size)
    h, z, mu, log_var = m.encode(packed_padded)


def test_loss(language, dataset):
    m = VAE(language.vocabulary_size)
    batch_size = 3
    packed_padded, padded = dataset.get_batch(0, batch_size)
    det_elbo(m, packed_padded, 1., test_loss=True)

