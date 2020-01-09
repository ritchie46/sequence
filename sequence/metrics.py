import numpy as np


def rank_scores(pred, target, k=20, reduction="average", include_p_at_k=True, include_mrr=True):
    """
    Precision @ k and Mean reciprocal rank.
    Modified from:
        https://github.com/CRIPAC-DIG/SR-GNN/blob/90123c88850eec8c574518fee6e46aefb42acb94/pytorch_code/model.py#L123

    TODO: Optimize. Broadcasting membership?

    Parameters
    ----------
    pred : torch.tensor
        Activations per session.
        Shape: (batch, seq_len, vocabulary)
    target : torch.tensor
        Shape: (batch, seq_len)
    k : int
    reduction : str
        'average' | 'sum'

    Returns
    -------
    p@k, mrr : tuple[float]

    """
    hits = []
    mrr = []
    pred_top_idx = pred.topk(k)[1]

    batch_size = target.shape[0]
    # Loop over batch
    for i in range(batch_size):
        # l
        seq = target[i, ...]

        # l, k
        pred = pred_top_idx[i]

        # Loop over sequence
        for j in range(min(len(seq), pred.shape[0])):
            idx = np.where(pred[j] == seq[j])[0]

            if include_p_at_k:
                hits.append(len(idx) > 0)
            if include_mrr:
                if len(idx) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (idx[0] + 1))

    if reduction == "average":
        return np.mean(hits), np.mean(mrr)

    return np.sum(hits), np.sum(mrr)

