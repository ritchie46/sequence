import numpy as np


def p_at_k(pred, target, k=20):
    target = target.cpu().data.numpy().T

    # shape: (batch, k)
    # numpys sort is faster
    pred_top = np.argsort(axis=0, a=pred.cpu().data.numpy())[::-1][:k, ...].squeeze().T
    # EOS to a non existing index
    pred_top[pred_top == 0] = -999

    c = 0
    for i in range(target.shape[0]):
        c += len(np.intersect1d(pred_top[i, :], np.unique(target[i, :])))

    return c / target.shape[0]
