import numpy as np


def rank_scores(
    pred, target, k=20, reduction="average", include_p_at_k=True, include_mrr=True
):
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
    include_p_at_k : bool
        Return precision @ k
    include_mrr : bool
        Return mean reciprocal rank

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
            # Stop at end of sequence
            if seq[j] == 0:
                break
            idx = np.where(pred[j] == seq[j])[0]

            if include_p_at_k:
                hits.append(len(idx) > 0)
            if include_mrr:
                if len(idx) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (idx[0] + 1))

    if reduction == "average":

        return (
            np.mean(hits) if include_p_at_k else None,
            np.mean(mrr) if include_mrr else None,
        )

    return (
        np.sum(hits) if include_p_at_k else None,
        np.sum(mrr) if include_mrr else None,
    )


def p_at_k(*args, **kwargs):
    return rank_scores(*args, **kwargs, include_mrr=False)[0]


def mrr(*args, **kwargs):
    return rank_scores(*args, **kwargs, include_p_at_k=False)[1]
