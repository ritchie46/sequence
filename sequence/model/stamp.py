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
    ):
        super().__init__(vocabulary_size, embedding_dim, custom_embeddings)

        if nonlinearity == "tanh":
            nl = nn.Tanh
        elif nonlinearity == "relu":
            nl = nn.ReLU
        else:
            raise ValueError(f"nonlinearity {nonlinearity} is not possible.")

        self.mlp_a = nn.Sequential(
            nn.Linear(self.embedding_dim, embedding_dim), nn.Tanh()
        )
        self.mlp_b = nn.Sequential(
            nn.Linear(self.embedding_dim, embedding_dim, bias=False), nn.Tanh()
        )

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
        padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(emb, padding_value=0)

        # l,b,e
        cumsum = torch.cumsum(padded, 0)
        batch_size = cumsum.shape[1]

        # Note that for the shorter sequences the cum avg is not correct
        # after the last time step of that sequence.
        # This is corrected in the loss calculation.
        lengths = torch.arange(1, 1 + cumsum.shape[0])
        return cumsum / lengths.reshape(cumsum.shape[0], 1, 1), padded

    def forward(self, x):
        # packed padded
        emb = self.apply_emb(x)

        # Rolling average session:
        #   m_s: l,b,e
        m_s, zero_padded_emb = self.external_memory(emb)

        # b,e
        h_s = self.mlp_a(m_s)

        # l,b,e
        h_t = self.mlp_b(zero_padded_emb)

        # v,e
        x = self.emb.weight

        # b,l,v
        z = torch.sigmoid(trilinear_composition(h_s, h_t, x))
        # Softmax over v
        return torch.log_softmax(z, -1)


def det_loss(model, packed_padded):
    y_hat = model(packed_padded)

    padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        packed_padded, padding_value=-1
    )

    return F.nll_loss(
        y_hat.reshape(-1, y_hat.shape[-1]), padded.T.flatten(), ignore_index=-1
    )
