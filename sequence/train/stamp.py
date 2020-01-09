import logging
from sequence.model.STAMP import det_loss
import torch
from sequence.utils import backward

logger = logging.getLogger(__name__)


def run_epoch(
    epoch,
    model,
    optim,
    dataset,
    batch_size,
    device="cpu",
    tensorboard_writer=None,
    global_step=0,
    n_batches=None,
    callbacks=[],
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
    global_step : int
        Step counter
    n_batches : int
        How many batches to run before epoch is regarded complete.
        Default = N / batch_size
    callbacks : list[function[**kwargs]]
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
        loss = det_loss(model, packed_padded)
        backward(loss, optim)
        optim.step()

        if global_step % 10 == 0:
            logger.info(
                "{}/{}\t{}%\tEpoch: {} Loss: {:.4f}".format(
                    c, n_batches, int(c / n_batches * 100), epoch, loss.item()
                )
            )

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss", loss.item(), global_step)

        [
            f(global_step=global_step, loss=loss, epoch=epoch, model=model,)
            for f in callbacks
        ]

    logger.debug("Epoch: {}\tLoss{:.4f}".format(epoch, loss.item()))
    return global_step
