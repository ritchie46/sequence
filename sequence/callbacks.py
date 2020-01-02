import os


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
