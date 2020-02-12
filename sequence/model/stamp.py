from torch import nn
import torch.nn.functional as F
import torch
from sequence.model.modular import Embedding
import numpy as np


def trilinear_composition(h_s, h_t, x, einsum=True):
    """
    Trilinear composition as described in:
    STAMP: Short-Term Attention/Memory Priority Model forSession-based Recommendation

    Shapes:
    b: batch
    e: embedding
    l: longest sequence length
    v: vocabulary size

    Parameters
    ----------
    h_s : torch.tensor
        Shape: (l, b, e)
    h_t : torch.tensor
        Shape: (l, b, e)
    x : torch.tensor
        Embedding matrix.
        Shape: (v, e)
    einsum : bool
     Use einsum. This is recommended as it appeared to be faster in profiling.
     The other option was added for testing purposes.

    Returns
    -------
    <h_s, h_t, x> : torch.tensor
        Shape: (b, l, v)
    """
    if einsum:
        return torch.einsum("lbe,lbe,ve->blv", h_s, h_t, x)

    else:
        b = h_s.shape[1]

        # h_t ⊙ x
        # l,b,e ⊙ v,1,1,e  (so b,e dimensions are broadcasted)
        bc = h_t * x.reshape(x.shape[0], 1, 1, x.shape[1])

        # aT @ (bc) = h_sT @ (h_t ⊙ x)
        # ---------
        # l,b,e @ v,l,e,b
        # b,e @ e,b will be matrix multiplied. v,l are untouched.
        # result is a v,l,b,b matrix
        vlbb = torch.matmul(h_s, bc.transpose(-1, -2))

        # As dot products should not be applied across batches, the diagonal can be selected
        # v,l,b
        vlb = vlbb[..., torch.arange(0, b), torch.arange(0, b)]
        # b,l,v
        return vlb.transpose(0, 2)


def cumavg(a):
    """

    Parameters
    ----------
    a : torch.tensor
        Shape: l,b,e

    Returns
    -------
    Cumulative average : torch.tensor
        Shape : l,b,e
        avg over l dimension
    """
    # l,b,e
    cumsum = torch.cumsum(a, 0)
    batch_size = cumsum.shape[1]

    # Note that for the shorter sequences the cum avg is not correct
    # after the last time step of that sequence.
    # This is corrected in the loss calculation.
    lengths = torch.arange(1, 1 + cumsum.shape[0], device=a.device)

    return cumsum / lengths.reshape(cumsum.shape[0], 1, 1)


class STMP(Embedding):
    """
    See Also:
        https://dl.acm.org/doi/pdf/10.1145/3219819.3219950?download=true
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dim=10,
        custom_embeddings=None,
        nonlinearity="tanh",
        mlp_layers=1,
    ):
        super().__init__(vocabulary_size, embedding_dim, custom_embeddings)

        if nonlinearity == "tanh":
            nl = nn.Tanh
        elif nonlinearity == "relu":
            nl = nn.ReLU
        else:
            raise ValueError(f"nonlinearity {nonlinearity} is not possible.")

        def gen_mlp(mlp_layer, nl):
            return nn.Sequential(
                *[
                    a
                    for _ in range(mlp_layers)
                    for a in (nn.Linear(self.embedding_dim, self.embedding_dim), nl())
                ]
            )

        self.mlp_a = gen_mlp(mlp_layers, nl)
        self.mlp_b = gen_mlp(mlp_layers, nl)

    def external_memory(self, emb):
        """
        Compute the cumulative average.

        In the paper, noted as m_s

        emb : torch.nn.utils.rnn.pack_padded_sequence

        Returns
        -------
        avg, : torch.tensor
            Average of the different sessions in the batch
            Shape: (l,b,e)
        padded : torch.tensor
            Zero padded embedding sequence
            Shape: (l,b,e)
        lengths : torch.tensor
            Lengths of the sequences in the batch.
            Shape: (batch,)

        """
        # Pad with zeros, so they don't influence the sum
        # Padded shape: (l,b,e)
        padded_emb, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            emb, padding_value=0
        )

        avg = cumavg(padded_emb)
        # m_s, m_t
        return avg, padded_emb

    def m_s_m_t(self, x):
        # packed padded
        emb = self.apply_emb(x)

        # Rolling average session:
        #   m_s: l,b,e
        return self.external_memory(emb)

    def _apply_from_m(self, m_s, m_t, return_all):
        # b,e
        h_s = self.mlp_a(m_s)

        # l,b,e
        h_t = self.mlp_b(m_t)

        # v,e
        x = self.emb.weight

        # b,l,v
        # Sigmoid is only for a single index!!
        # Applying the sigmoid saturates the softmax. Leading to poor results.
        # Paper should have taken another notation!
        z = trilinear_composition(h_s, h_t, x)

        if return_all:
            # packed padded
            return m_s, m_t, h_s, h_t, x, z, torch.log_softmax(z, -1)

        # Softmax over v
        return torch.log_softmax(z, -1)

    def forward(self, x, return_all=False):
        # Rolling average session:
        #   m_s: l,b,e
        m_s, m_t = self.m_s_m_t(x)
        return self._apply_from_m(m_s, m_t, return_all)


class STAMP(STMP):
    def __init__(
        self,
        vocabulary_size,
        embedding_dim=10,
        custom_embeddings=None,
        nonlinearity="tanh",
        mlp_layers=1,
    ):
        super().__init__(
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            custom_embeddings=custom_embeddings,
            nonlinearity=nonlinearity,
            mlp_layers=mlp_layers,
        )
        self.attention_net = AttentionNet(self.embedding_dim)

    def forward(self, x, return_all=False):
        _, m_t = self.m_s_m_t(x)
        m_s = self.attention_net(m_t)
        return self._apply_from_m(m_s, m_t, return_all)


class AttentionNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        # Only W3 has bias. Its bias serves as b_a
        self.w1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w3 = nn.Linear(embedding_dim, embedding_dim)
        self.w0 = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, emb, return_attention_factors=False):
        """

        Parameters
        ----------
        emb : torch.tensor
            Shape: l,b,e
        m_s : torch.tensor
            Shape: l,b,e

        Returns
        -------

        """
        # loop over lengths
        # So results for whole batch
        m_a = []
        a_is = []
        for i in range(emb.shape[0]):
            # one,b,e
            x_t = emb[i, ...].unsqueeze(0)

            # i + 1, b, e
            x_i = emb[: i + 1, ...]

            m_s_ = x_i.sum(0).unsqueeze(0)

            # i + 1, b, e
            # Attention factors. The lenth of this tensor
            # changes dependend on current t
            a_i = self.w0(torch.sigmoid(self.w1(x_i) + self.w2(x_t) + self.w3(m_s_)))

            if return_attention_factors:
                a_is.append(a_i)

            # one, b, e
            m_a.append(torch.sum(a_i * x_i, 0).unsqueeze(0))

        # l, b, e
        m_a = torch.cat(m_a, dim=0)

        if return_attention_factors:
            return m_a, a_is

        return m_a


def det_loss(
    model, packed_padded, test_loss=False, scale_loss_by_lengths=True, max_len=1.0
):
    # b,l,v
    y_hat = model(packed_padded)

    # padded: l,b
    padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        packed_padded, padding_value=-1
    )
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

        loss_ = loss_.sum() / batch_size
        assert np.allclose(loss_.item(), loss.item())

    if not scale_loss_by_lengths:
        loss = loss / (target.shape[0] * lengths.to(loss.dtype).mean()) * max_len * 100

    return loss / target.shape[0]
