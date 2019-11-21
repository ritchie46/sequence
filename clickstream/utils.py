import torch


def elementwise_apply(fn, *args):
    """
    Apply elementwise function over packed tensors.

    Parameters
    ----------
    fn : function
        Can be for instance a torch.nn.Embedding instance
    args : sequence

    Returns
    -------
    out : [fn(x), fn(x)]

    """
    return torch.nn.utils.rnn.PackedSequence(
        fn(
            *[
                (arg.data if type(arg) == torch.nn.utils.rnn.PackedSequence else arg)
                for arg in args
            ]
        ),
        args[0].batch_sizes,
    )
