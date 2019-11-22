from torch import nn
import numpy as np
import torch.nn.functional as F
import torch

from clickstream.utils import masked_flip


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dim=10,
        latent_size=20,
        bidirectional=True,
        rnn_layers=1,
    ):
        super().__init__()
        if bidirectional:
            linear_in = latent_size * 2
        else:
            linear_in = latent_size

        self.vocabulary_size = vocabulary_size
        self.emb = nn.Embedding(vocabulary_size, embedding_dim)
        self.rnn_enc = nn.GRU(
            embedding_dim,
            latent_size,
            bidirectional=bidirectional,
            num_layers=rnn_layers,
        )

        # decoder
        self.rnn_dec = nn.GRU(
            embedding_dim,
            latent_size,
            bidirectional=bidirectional,
            num_layers=rnn_layers,
        )
        self.decoder_out = nn.Sequential(
            nn.Linear(linear_in, vocabulary_size), nn.LogSoftmax(1)
        )

    def encode(self, x):
        """

        Parameters
        ----------
        x : PackedPaddedSequence
            Shape: (seq_len, batch)

        Returns
        -------
        h : torch.tensor
            Last hidden state
            shape: (batch, input_size)
        """
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, padding_value=0)
            emb = self.emb(padded)
            emb = torch.nn.utils.rnn.pack_padded_sequence(
                emb, lengths, enforce_sorted=False
            )
        else:
            emb = self.emb(x).reshape(len(x), 1, -1)
        out, h = self.rnn_enc(emb)

        return h

    def decode(self, word=None, h=None):
        """
        word : tensor
            shape: (batch)
        h : tensor
            shape: (num_layers * num_directions, seq_len, batch, feat)
        """
        if isinstance(h, tuple):
            batch_size = h[0].shape[1]
            device = h[0].device
        else:
            batch_size = h.shape[1]
            device = h.device
        if word is None:
            # (seq_len, batch)
            word = torch.ones(1, batch_size, device=device, dtype=torch.long)
        else:
            word = word.unsqueeze(0)

        # (seq_len, batch, embedding)
        emb = self.emb(word)
        out, h = self.rnn_dec(emb, h)

        if batch_size > 1:
            out = out.squeeze()
        return self.decoder_out(out.reshape(batch_size, -1)), h


def decoder_loss_batched(
    model,
    packed_padded,
    teach_forcing_p=0.5,
    nullify_rnn_input=False,
    reverse_target=False,
):
    loss = 0
    w = None
    h = model.encode(packed_padded)
    padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        packed_padded, padding_value=-1
    )
    if reverse_target:
        padded = masked_flip(padded.T, lengths)

        # remove EOS token
        padded = padded[1:, :]

    # Loop is over the words in a sequence. Not over the batch
    for i in range(padded.shape[0]):
        target = padded[i, :]
        out, h = model.decode(word=w, h=h)

        # ignore the targets that are -1 (this is the padding value)
        loss_ = F.nll_loss(out, target, ignore_index=-1, reduction="none")
        loss += (loss_ / lengths.to(loss_.device)).sum()

        if nullify_rnn_input:
            w = None
        elif np.random.rand() < teach_forcing_p:
            # make new tensor, otherwise the computation graph breaks due to inplace operation
            w = target.clone().detach()
            w[w < 0] = 0
        else:
            # new word
            w = out.argmax(1).detach()
    return loss


def decoder_loss(m, padded):
    loss = 0
    for i in range(padded.shape[1]):
        target = padded[:, i]
        target = target[target >= 0]

        z = m.encode(target)
        sen_loss = 0
        for j in range(len(target)):
            out, z = m.decode(word=None, h=z)
            sen_loss += F.nll_loss(out.reshape(1, -1), target[j].unsqueeze(0))

        loss += sen_loss / len(target)

    return loss


def run_decoder(m, h, padded, nullify_rnn_input=False):
    w = None
    words = []
    for i in range(padded.shape[0]):
        out, h = m.decode(word=w, h=h)
        # new word
        w = out.argmax(1)
        words.append(w.unsqueeze(0))
        if nullify_rnn_input:
            w = None
    return torch.cat(words, dim=0).T
