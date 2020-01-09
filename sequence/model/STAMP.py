from torch import nn
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
        Shape: (b, e)
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
        return torch.einsum("be,lbe,ve->blv", h_s, h_t, x)

    else:
        b = h_s.shape[0]

        # h_t ⊙ x
        # l,b,e ⊙ v,1,1,e  (so b,e dimensions are broadcasted)
        bc = h_t * x.reshape(x.shape[0], 1, 1, x.shape[1])

        # aT @ (bc)
        # ---------
        # b,e @ v,l,e,b
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

    def __init__(self, vocabulary_size, embedding_dim=10, custom_embeddings=None):
        super().__init__(vocabulary_size, embedding_dim, custom_embeddings)
        self.mlp_a = nn.Sequential(
            nn.Linear(self.embedding_dim, embedding_dim),
            # nn.ReLU()
        )
        self.mlp_b = nn.Sequential(
            nn.Linear(self.embedding_dim, embedding_dim, bias=False),
            # nn.ReLU()
        )

    def external_memory(self, emb):
        """
        This is simply the average of the session sequence.
        In the paper, noted as m_s

        emb : torch.nn.utils.rnn.pack_padded_sequence

        Returns
        -------
        avg, : torch.tensor
            Average of the different sessions in the batch
            Shape: (batch, embedding)
        padded : torch.tensor
            Zero padded embedding sequence
            Shape: (longest_sequence, batch, embedding)
        lengths : torch.tensor
            Lengths of the sequences in the batch.
            Shape: (batch,)

        """
        # Pad with zeros, so they don't influence the sum
        # Padded shape: (longest_sequence, batch, embedding)
        padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(emb, padding_value=0)

        # Shape: (batch, embedding)
        sum_ = torch.sum(padded, 0)
        batch_size = sum_.shape[0]

        return sum_ / lengths.reshape(batch_size, -1), padded, lengths

    def forward(self, x):
        # packed padded
        emb = self.apply_emb(x)
        m_s, zero_padded_emb, seq_lengths = self.external_memory(emb)

        # b,e
        h_s = self.mlp_a(m_s)

        # l,b,e
        h_t = self.mlp_b(zero_padded_emb)

        # v,e
        x = self.emb.weight

        trilinear_composition(h_s, h_t, x)
