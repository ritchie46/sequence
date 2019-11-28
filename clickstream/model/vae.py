import torch
from torch import nn

from clickstream.model.seq2seq import EncoderDecoder


class VAE(EncoderDecoder):
    def __init__(
        self,
        vocabulary_size,
        embedding_dim=10,
        hidden_size=20,
        latent_size=20,
        bidirectional=True,
        rnn_layers=1,
        custom_embeddings=None,
    ):
        super(VAE, self).__init__(
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            latent_size=hidden_size,  # In the AE, latent and hidden are equal.
            bidirectional=bidirectional,
            rnn_layers=rnn_layers,
            custom_embeddings=custom_embeddings,
            rnn_type='gru',
        )
        self.latent_size = latent_size
        # First part of the output vector is mu, second is sigma
        self.variational_params = nn.Linear(self.linear_in, latent_size * 2)
        self.latent2hidden = nn.Linear(latent_size, self.linear_in)

    def reparameterize(self, h):
        """

        Parameters
        ----------
        h : torch.tensor
            Flattened hidden state
            Shape: (batch_size, num_layers * num_directions * feat)

        Returns
        -------
        z : torch.tensor
            Shape: (batch_size, latent_size)

        """
        p = self.variational_params(h)
        mu = p[:, :self.latent_size]
        log_var = p[:, self.latent_size:]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, log_var

    def encode(self, x):
        h = super().encode(x)
        hidden_shape = h.shape
        batch_size = h.shape[1]

        # flatten h
        h = h.reshape(batch_size, -1)
        z, mu, log_var = self.reparameterize(h)

        # map latent to RNN hidden state
        h = self.latent2hidden(z).reshape(hidden_shape)
        return h, z, mu, log_var



