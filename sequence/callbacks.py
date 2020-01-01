import os


def save_every_n_steps(n, mr=None, dump_dir="artifacts"):
    os.makedirs(dump_dir, exist_ok=True)

    def callback(**kwargs):
        step = kwargs["global_step"]
        if step % n == 0:
            with open(
                os.path.join(dump_dir, f"epoch-{kwargs['epoch']}_step-{step}.pkl"), "wb"
            ) as f:
                mr.dump(f)

    return callback
