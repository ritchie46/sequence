import numpy as np


def p_at_k(pred, target, k=20):
    """

    Parameters
    ----------
    pred : Union[torch.Tensor, np.array]
        Shape: (seq_len, batch)
    target : Union[torch.Tensor, np.array]
        Shape: (seq_len, batch)
    k : int

    Returns
    -------

    """
    if hasattr(target, 'cpu'):
        target = target.cpu().data.numpy().T
    else:
        target = target.T

    if hasattr(target, 'cpu'):
        pred = pred.cpu().data.numpy()

    # shape: (batch, k)
    # numpys sort is faster
    pred_top = np.argsort(axis=0, a=pred)[::-1][:k, ...].squeeze().T
    # EOS to a non existing index
    pred_top[pred_top == 0] = -999

    c = 0
    for i in range(target.shape[0]):
        c += len(np.intersect1d(pred_top[i, :], np.unique(target[i, :])))

    return c / target.shape[0]
