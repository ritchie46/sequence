import os
from sequence import metrics
import torch
import numpy as np


def apply(callbacks, **kwargs):
    """
    global_step : int
    loss : float
    epoch : int
    model :
    ds_train :
    ds_test :
    logger :
    device : str
    epoch_p : float
    tensorboard_writer :
        Percentage done of epoch
    """
    [f(**kwargs) for f in callbacks]


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
        if kwargs["epoch_p"] != 0:
            return
        model = kwargs["model"]
        logger = kwargs["logger"]
        device = kwargs["device"]

        for name, ds in zip(["train", "test"], [kwargs["ds_train"], kwargs["ds_test"]]):
            p_at_k_ = []
            mrr_ = []
            for i in range(n):
                packed_padded, padded = ds.get_batch(
                    i * 100, i * 100 + 100, device=device
                )
                if model.__class__.__name__ == "LSTM":
                    state_h, state_c = model.init_state(padded.shape[1])

                with torch.no_grad():
                    if model.__class__.__name__ == "LSTM":
                        pred, _ = model(packed_padded, (state_h, state_c))
                    else:
                        pred = model(packed_padded)

                target = padded.T[:, 1:]
                p_at_k, mrr = metrics.rank_scores(
                    pred.cpu(), target.cpu(), k=k, skip_first_k=0
                )
                p_at_k_.append(p_at_k)
                mrr_.append(mrr)

            p_at_k = np.mean(p_at_k_)
            mrr = np.mean(mrr_)
            logger.info("P@{}: {:.3f}, MRR: {:.3f}".format(k, p_at_k, mrr))
            if kwargs["tensorboard_writer"]:
                kwargs["tensorboard_writer"].add_scalar(
                    f"p_at_k_{name}_{k}", p_at_k, kwargs.get("global_step", 0)
                )
                kwargs["tensorboard_writer"].add_scalar(
                    f"MRR_at_{name}_{k}", mrr, kwargs.get("global_step", 0)
                )

    return callback
