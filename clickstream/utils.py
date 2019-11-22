import torch


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