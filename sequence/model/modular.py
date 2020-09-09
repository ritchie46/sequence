from torch import nn
import torch
from typing import Union


class Embedding(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 20,
        custom_embeddings: Union[None, torch.FloatTensor] = None,
    ):
        super().__init__()

        if custom_embeddings is None:
            self.emb = nn.Embedding(vocabulary_size, embedding_dim)
            self.vocabulary_size = vocabulary_size
            self.embedding_dim = embedding_dim
        else:
            self.vocabulary_size = custom_embeddings.shape[0]
            self.embedding_dim = custom_embeddings.shape[1]
            self.emb = nn.Embedding(*custom_embeddings.shape, _weight=custom_embeddings)  # type: ignore
            self.emb.weight.requires_grad = True

    def apply_emb(self, x, pack=False):
        """
        Parameters
        ----------
        x : Union[PackedPaddedSequence, PaddedTensor]
            Shape: (seq_len, batch)
        pack : bool
            Return PackedPaddedSequence. If input type of x == PackedPaddedSequence
            output always is PackedPaddedSequence

        Returns
        -------
        embedding : Union[PackedPaddedSequence, PaddedTensor]

        """
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
                lengths = torch.full(
                    (x.shape[1],), fill_value=x.shape[0], device=x.device
                )

        if pack:
            emb = torch.nn.utils.rnn.pack_padded_sequence(
                emb, lengths, enforce_sorted=False
            )
        return emb
