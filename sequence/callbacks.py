import os
from sequence import metrics
import torch
import numpy as np


def apply(
    callbacks,
    global_step=None,
    loss=None,
    epoch=None,
    model=None,
    ds_train=None,
    ds_test=None,
    logger=None,
    device=None,
    epoch_p=None,
):
    """
    epoch_p : float
        Percentage done of epoch
    """
    [
        f(
            global_step=global_step,
            loss=loss,
            epoch=epoch,
            model=model,
            ds_train=ds_train,
            ds_test=ds_test,
            logger=logger,
            device=device,
            epoch_p=epoch_p
        )
        for f in callbacks
    ]


def save_every_n_steps(n, mr=None, dump_dir="artifacts"):
    """
    Save model every n iterations.

    Parameters
    ----------
    n : int
        Save every n intervals.
    mr : ModelRegistry
    dump_dir : str
        Path to save pickled files.
    """
    os.makedirs(dump_dir, exist_ok=True)

    def callback(**kwargs):
        step = kwargs["global_step"]
        if step % n == 0:
            with open(
                os.path.join(dump_dir, f"epoch-{kwargs['epoch']}_step-{step}.pkl"), "wb"
            ) as f:
                mr.dump(f)

    return callback


def register_global_step(mr):
    """
    Add global step to ModelRegistry.

    Parameters
    ----------
    mr : ModelRegistry
    """

    def callback(**kwargs):
        mr.global_step_ = kwargs["global_step"]

    return callback


def log_ranking_metrics(n, k):
    def callback(**kwargs):
        if kwargs['epoch_p'] != 0:
            return
        model = kwargs["model"]
        ds = kwargs["ds_train"]
        logger = kwargs["logger"]
        device = kwargs["device"]

        p_at_k_ = []
        mrr_ = []
        for i in range(n):
            packed_padded, padded = ds.get_batch(i * 100, i * 100 + 100, device=device)

            with torch.no_grad():
                pred = model(packed_padded)

            target = padded.T[:, 1:]
            p_at_k, mrr = metrics.rank_scores(
                pred.cpu(), target.cpu(), k=k, skip_first_k=0
            )
            p_at_k_.append(p_at_k)
            mrr_.append(mrr)

        logger.info(
            "P@{}: {:.3f}, MRR: {:.3f}".format(k, np.mean(p_at_k_), np.mean(mrr_))
        )

    return callback
