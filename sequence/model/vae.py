import torch
from torch import nn
import torch.nn.functional as F
from sequence.model.seq2seq import EncoderDecoder
import numpy as np


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
            rnn_type="gru",
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
        mu = p[:, : self.latent_size]
        log_var = p[:, self.latent_size :]
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

    def decode(self, x, h):
        emb = self.apply_emb(x)
        packed_padded_out, h = self.rnn_dec(emb, h)

        # padded, Shape: (batch_size, seq_len, num_directions * n_layers)
        padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_padded_out, padding_value=-1, batch_first=True
        )
        # (batch_size, seq_len, vocabulary_size)
        out = self.decoder_out(padded)
        return out


def run_decoder(model, packed_padded, word_dropout, h):
    padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        packed_padded, padding_value=-1
    )
    # Decoder can also be done in a single batch, as we  don't feed the output
    # of the rnn back into itself. We only pass the correct words, or unknown.
    padded_decoder = padded.clone().detach()
    if word_dropout > 0:
        mask = padded > 0
        keep_prob = torch.bernoulli(
            torch.full_like(mask, word_dropout, dtype=torch.float)
        )
        keep_prob[padded_decoder < 0] = 1.0

        # 2 is the "UNKNOWN" word
        mask = (mask * keep_prob).bool()
        padded_decoder = torch.masked_fill(padded_decoder, mask, 2)

    packed_padded_decoder = torch.nn.utils.rnn.pack_padded_sequence(
        padded_decoder, lengths, enforce_sorted=False
    )
    out = model.decode(packed_padded_decoder, h)
    target = padded.T

    return out, target


def det_neg_elbo(
    model,
    packed_padded,
    word_dropout=1.0,
    test_loss=False,
):
    # https://arxiv.org/pdf/1511.06349.pdf
    h, z, mu, log_var = model.encode(packed_padded)
    out, target = run_decoder(model, packed_padded, word_dropout, h)

    nll = F.nll_loss(
        out.reshape(-1, out.shape[-1]),
        target.flatten(),
        ignore_index=-1,
        reduction="sum",
    )

    # loop over sequences
    if test_loss:
        loss_ = 0
        for i in range(target.shape[1]):
            loss_ += F.nll_loss(
                out[:, i, :], target[:, i], ignore_index=-1, reduction="sum"
            )
        assert np.allclose(loss_.item(), nll.item())

    kl = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
    return nll, kl


