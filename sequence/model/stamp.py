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
        z = torch.sigmoid(trilinear_composition(h_s, h_t, x))

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
        m_s, m_t = self.m_s_m_t(x)
        m_s = self.attention_net(m_t, m_s)
        return self._apply_from_m(m_s, m_t, return_all)



class AttentionNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        # Only W3 has bias. Its bias serves as b_a
        self.w1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w3 = nn.Linear(embedding_dim, embedding_dim)
        self.w0 = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, emb, m_s):
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
        # l,b,1
        a = self.w0(torch.sigmoid(self.w1(emb) + self.w2(emb) + self.w3(m_s)))
        # l,b,e
        applied = a * emb

        return torch.cumsum(applied, 0)


def det_loss(model, packed_padded):
    # b,l,v
    y_hat = model(packed_padded)

    # padded: l,b
    padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        packed_padded, padding_value=-1
    )
    # The prediction and the target need to be offset.
    # input x_0:x_t prediction should be x_t+1
    # Therefore remove last prediction t_n, and remove first target t0.
    y_hat = y_hat[:, :-1, :]

    target = padded.T[:, 1:]
    target[target == 0] = -1

    return F.nll_loss(
        y_hat.reshape(-1, y_hat.shape[-1]), target.reshape(-1), ignore_index=-1
    )
