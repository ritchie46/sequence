import torch
import os
from logging import getLogger
import numpy as np

logger = getLogger(__name__)
try:
    from apex import amp

    logger.info("Mixed Precision training possible!")
except ImportError:
    logger.info("Could not import apex. Mixed Precision training is not possible.")


def backward(loss, optim):
    if os.environ.get("USE_APEX_AMP"):
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()


def masked_flip(padded_sequence, sequence_lengths):
    """
        Flips a padded tensor along the time dimension without affecting masked entries.

        Source: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L376-L398

        Parameters
        ----------
        padded_sequence : torch.tensor
            The tensor to flip along the sequence dimension.
            Shape: (batch, seq_len)
        sequence_lengths : torch.tensor
            A list containing the lengths of each unpadded sequence in the batch.
        Returns
        -------
        A ``torch.Tensor`` of the same shape as padded_sequence.
    """
    assert padded_sequence.size(0) == len(
        sequence_lengths
    ), f"sequence_lengths length ${len(sequence_lengths)} does not match batch size ${padded_sequence.size(0)}"

    seq_len = padded_sequence.shape[1]
    flipped_padded_sequence = torch.flip(padded_sequence, dims=[1])

    sequences = [
        # shape: (batch, seq_len)
        flipped_padded_sequence[i, seq_len - length :]
        for i, length in enumerate(sequence_lengths)
    ]
    return torch.nn.utils.rnn.pad_sequence(sequences, padding_value=-1)


def get_batch_size(h):
    if isinstance(h, tuple):
        return h[0].shape[1]
    return h.shape[1]


def anneal(i, goal, f="linear", a=2.0):
    if f == "linear":
        return min(1.0, i / goal)
    else:
        return min(1.0, (i / goal) ** a)


# Annealing functions courtesy of fast ai
# https://github.com/fastai/fastai/blob/cc5abce71be67b2873003568526208f39cbc616b/fastai/callback.py#L355
def annealing_no(start, end, pct):
    return start


def annealing_linear(start, end, pct):
    return start + pct * (end - start)


def annealing_cosine(start, end, pct):
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


def annealing_sigmoid(start, end, pct):
    return min(annealing_cosine(start, end, pct), 1.0)


def annealing_exp(start, end, pct):
    v = pct * (end - start) + start
    return min(1.0, (v / end) ** 2)


def translate_sentence_i2w(language, sentence):
    # evaluate property only once
    d = language.i2w
    return [d[int(j)] for j in sentence]
