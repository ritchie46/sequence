import torch
import torch.nn.functional as F

from sequence.model.modular import Embedding

from torch import nn, optim

import pickle
import numpy as np


class LSTM(Embedding):
    def __init__(self, vocabulary_size, embedding_dim=10, custom_embeddings=None):
        self.lstm_layers = 1
        self.latent_size = 20
        super().__init__(vocabulary_size, embedding_dim, custom_embeddings)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.latent_size,
            num_layers=self.lstm_layers,
            # dropout=0.2,
            batch_first=True,
        )
        self.fc = nn.Linear(self.latent_size, vocabulary_size)

    def forward(self, x, prev_state):
        emb = self.apply_emb(x)
        padded_emb, lengths = nn.utils.rnn.pad_packed_sequence(
            emb, padding_value=0, batch_first=True
        )
        output, state = self.lstm(padded_emb, prev_state)
        fc = self.fc(output)

        return torch.log_softmax(fc, -1), state

    def init_state(self, batch):
        # TODO use random number for init

        return (
            torch.zeros(self.lstm_layers, batch, self.latent_size),
            torch.zeros(self.lstm_layers, batch, self.latent_size),
        )


def det_loss(
    model: LSTM,
    packed_padded: nn.utils.rnn.PackedSequence,
    test_loss: bool = False,
    scale_loss_by_lengths: bool = False,
    max_len: int = 1,
    state_h: tuple = None,
    state_c: tuple = None,
):
    # b,l,v
    y_hat, (state_h, state_c) = model(packed_padded, (state_h, state_c))

    # padded: l,b
    padded, lengths = nn.utils.rnn.pad_packed_sequence(packed_padded, padding_value=-1)
    lengths = lengths.to(device=padded.device)
    # The prediction and the target need to be offset.
    # input x_0:x_t prediction should be x_t+1
    # Therefore remove last prediction t_n, and remove first target t0.
    y_hat = y_hat[:, :-1, :]

    # b, l
    target = padded.T[:, 1:]
    target[target == 0] = -1
    # -1 for the offset of 1
    # -1 for mapping 0 to -1
    lengths -= 2
    batch_size = target.shape[0]

    loss = F.nll_loss(
        y_hat.reshape(-1, y_hat.shape[-1]),
        target.reshape(-1),
        ignore_index=-1,
        reduction="none",
    )

    # Divide every sequence loss by the length of the sequence
    if scale_loss_by_lengths:
        loss = loss.reshape(target.shape) / lengths.reshape(-1, 1)

    # Divide the overall loss by the batch size
    loss = loss.sum() / target.shape[0]

    if test_loss:
        loss_ = 0

        # loop over sequences
        for i in range(target.shape[1] - 1):
            loss_ += (
                F.nll_loss(
                    y_hat[:, i, :], target[:, i], ignore_index=-1, reduction="none"
                )
                / lengths
            )

        # loss is a vector because of vectorization
        loss_ = loss_.sum() / batch_size  # type: ignore

        assert np.allclose(loss_.item(), loss.item())  # type: ignore

    if not scale_loss_by_lengths:
        loss = loss / (target.shape[0] * lengths.to(loss.dtype).mean()) * max_len * 100

    return loss / target.shape[0]
