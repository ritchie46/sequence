import numpy as np
from dask.array.core import Array


def p_at_k(pred, target, k=20, reduction="average"):
    """

    Parameters
    ----------
    pred : Union[torch.Tensor, np.array]
        Activations per session.
        Shape: (batch, seq_len)
    target : Union[torch.Tensor, np.array]
        Shape: (batch, seq_len)
    k : int
    reduction : str
        'average' | 'sum'

    Returns
    -------
    p@k : float

    """
    if hasattr(target, "cpu"):
        target = target.cpu().data.numpy()
    if isinstance(target, Array):
        target = np.array(target)

    if hasattr(pred, "cpu"):
        pred = pred.cpu().data.numpy()

    # shape: (batch, k)
    # numpys sort is faster
    pred_top = np.argsort(axis=-1, a=pred)[..., ::-1][..., :k]

    # EOS to a non existing index
    pred_top[pred_top == 0] = -999

    c = 0
    for i in range(target.shape[0]):
        c += len(np.intersect1d(pred_top[i, :], np.unique(target[i, :])))

    if reduction == "average":
        c /= target.shape[0]

    return c
