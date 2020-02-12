import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch


def minmax(x):
    _min = x.min()
    return (x - _min) / (x.max() - _min)


def identity(x):
    return x


def get_attn_weights(model, packed_padded, min_max=False):
    """

    Parameters
    ----------
    model : sequence.model.stamp.STAMP
    packed_padded : torch.nn.utils.rnn.PackedSequence
        As returned by `sequence.data.utils.Dataset#get_batch`

    Returns
    -------
    ai : list[torch.Tensor]
        Length of the list is equal to sequence length -1 (as attention looks back)

        Every tensor in the list has shape `t, b`
        where `t` is a timestep incrementing by 1.
    """
    if not min_max:
        minmax = lambda x: x.abs() / x.abs().sum()

    with torch.no_grad():
        _, m_t = model.m_s_m_t(packed_padded)
        _, ai = model.attention_net(m_t, return_attention_factors=True)

    return list(map(lambda x: minmax(x.reshape(-1, x.shape[1])), ai))


def make_attn_plot(ai_s, timestep=1, batch=0):
    """
    Plot attention weights.

    Parameters
    ----------
    ai_s : torch.tensor
        Shape: `t, b`
        Attention factors
    timestep : int
        Timestep to select
    batch : int
        Batch to select

    Returns
    -------
    fig : matplotlib.figure.Figure

    """

    ai_s = ai_s[timestep][:, batch]
    fig = plt.figure(edgecolor="white", figsize=(len(ai_s) * 2, 0.5))
    ax = plt.gca()
    ax.axis("off")

    for i, ai in enumerate(ai_s):
        ax.add_patch(Rectangle(xy=[i, 0], width=1, height=1, alpha=ai, ec="black"))
        ax.text(i + 0.5, 0.5, "{:.2f}".format(float(ai)))
    ax.set_xlim(0, len(ai_s))
    ax.set_ylim(0, 1)

    return fig
