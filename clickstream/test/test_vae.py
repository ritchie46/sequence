from clickstream.model.vae import VAE
from clickstream.test.test_ae import language, words, dataset, paths


def test_flow(language, dataset):
    m = VAE(language.vocabulary_size)
    batch_size = 3

    packed_padded, padded = dataset.get_batch(0, batch_size)
    h, z, mu, log_var = m.encode(packed_padded)

