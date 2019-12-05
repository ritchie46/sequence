import logging
from sequence.model.vae import det_neg_elbo
import torch
from sequence.utils import backward

logger = logging.getLogger(__name__)


def run_epoch(
    epoch,
    model,
    optim,
    dataset,
    batch_size,
    word_dropout=0.8,
    device="cpu",
    tensorboard_writer=None,
    global_step=0,
    anneal_f=lambda x: 1,
    n_batches=None
):
    """
    Train one epoch.

    Parameters
    ----------
    epoch : int
    model : torch.nn.Module
    optim : torch.Optimizer
    dataset : sequence.data.utils.Dataset
    batch_size : int
    device : str
        "cpu" or "cuda"
    tensorboard_writer : SummaryWriter
    """
    model.train()
    n_total = len(dataset)
    dataset.shuffle()
    if n_batches is None:
        n_batches = n_total // batch_size

    c = 0
    for i in range(n_batches):
        global_step += 1
        c += 1
        optim.zero_grad()
        i = i * batch_size

        packed_padded, padded = dataset.get_batch(i, i + batch_size, device=device)
        _, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_padded, padding_value=-1
        )

        nll, kl = det_neg_elbo(model, packed_padded, word_dropout)
        anneal_factor = anneal_f(global_step)
        # Scale by lengths
        loss = (nll + kl * anneal_factor) / lengths.sum()
        backward(loss, optim)
        optim.step()

        if global_step % 10 == 0:
            logger.debug(
                "{}/{}\t{}%\tLoss: {:.4f}".format(
                    c, n_batches, int(c / n_batches * 100), loss.item()
                )
            )

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("NEG_ELBO", loss.item(), global_step)
                tensorboard_writer.add_scalars(
                    "ELBO_PARTS",
                    {"NLL": nll.item(), "KL": kl.item()},
                    global_step,
                )
                tensorboard_writer.add_scalar("Anneal_factor", anneal_factor, global_step)

    logger.debug("Epoch: {}\tLoss{:.4f}".format(epoch, loss.item()))
    return global_step