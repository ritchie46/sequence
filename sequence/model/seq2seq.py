from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from sequence.utils import masked_flip, get_batch_size


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dim=10,
        latent_size=20,
        bidirectional=True,
        rnn_layers=1,
        custom_embeddings=None,
        rnn_type="gru",
    ):
        """
        Seq2Seq encoder decoder model.

        Parameters
        ----------
        vocabulary_size : int
            Number of words in the vocabulary
        embedding_dim : int
            Size of the embeddings.
        latent_size : int
            Size of the hidden state vectors of the RNN.
        bidirectional : bool
            Bidirectional RNN.
        rnn_layers : int
            Number of RNN layers.
        custom_embeddings : torch.tensor
            Custom word embeddings, such as GLOVE or Word2Vec.
        rnn_type : str
            'gru' or 'lstm'
        """
        super().__init__()
        if bidirectional:
            self.linear_in = latent_size * 2
        else:
            self.linear_in = latent_size
        self.linear_in *= rnn_layers

        if custom_embeddings is None:
            self.emb = nn.Embedding(vocabulary_size, embedding_dim)
        else:
            vocabulary_size = custom_embeddings.shape[0]
            embedding_dim = custom_embeddings.shape[1]
            self.emb = nn.Embedding(*custom_embeddings.shape, _weight=custom_embeddings)
            self.emb.weight.requires_grad = False

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.vocabulary_size = vocabulary_size
        self.rnn_enc = rnn(
            embedding_dim,
            latent_size,
            bidirectional=bidirectional,
            num_layers=rnn_layers,
        )

        # decoder
        self.rnn_dec = rnn(
            embedding_dim,
            latent_size,
            bidirectional=bidirectional,
            num_layers=rnn_layers,
        )
        self.decoder_out = nn.Sequential(
            nn.Linear(self.linear_in, vocabulary_size), nn.LogSoftmax(-1)
        )

    def apply_emb(self, x, pack=False):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, padding_value=0)
            emb = self.emb(padded)
            pack = True
        else:
            if len(x.shape) > 1:
                shape = (x.shape[0], x.shape[1], -1)
            else:
                shape = (x.shape[0], 1, -1)
            emb = self.emb(x).reshape(*shape)

            if pack:
                lengths = torch.full((x.shape[1],), fill_value=x.shape[0], device=x.device)

        if pack:
            emb = torch.nn.utils.rnn.pack_padded_sequence(
                emb, lengths, enforce_sorted=False
            )
        return emb

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
        emb = self.apply_emb(x)
        out, h = self.rnn_enc(emb)

        return h

    def decode(self, word=None, h=None):
        """
        word : tensor
            shape: (batch)
        h : Union[tensor, tuple]
            Tensor:
                shape: (num_layers * num_directions, batch, feat)
            Tuple: (h, c)
                Both have the same shape as h.
        """
        if isinstance(h, tuple):
            batch_size = h[0].shape[1]
            device = h[0].device
        else:
            batch_size = h.shape[1]
            device = h.device
        if word is None:
            # (seq_len, batch)
            # UNKNOWN word
            word = torch.ones(1, batch_size, device=device, dtype=torch.long) * 2
        else:
            word = word.unsqueeze(0)

        # (seq_len, batch, embedding)
        emb = self.emb(word)
        out, h = self.rnn_dec(emb, h)

        if batch_size > 1:
            out = out.squeeze()
        return self.decoder_out(out.reshape(batch_size, -1)), h


def det_loss_batched(
    model,
    packed_padded,
    teach_forcing_p=0.5,
    nullify_rnn_input=False,
    reverse_target=False,
):
    """

    Parameters
    ----------
    model : sequence.model.seq2seq.EncoderDecoder
    packed_padded : torch.nn.utils.rnn.PackedSequence
    teach_forcing_p : float
    nullify_rnn_input : bool
    reverse_target : bool
        Predict A B, C --> C, B, A if True,
        Else; A, B, C --> A, B, C

    Returns
    -------
    loss : torch.tensor
    """

    loss = 0
    w = None
    h = model.encode(packed_padded)
    batch_size = get_batch_size(h)
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
    return loss / batch_size


def det_loss(m, padded):
    """

    Parameters
    ----------
    m : sequence.model.seq2seq.EncoderDecoder
    padded : torch.tensor
        -1 padded sequences

    Returns
    -------
    loss : torch.tensor
    """
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
