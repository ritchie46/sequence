import logging
from sequence.model.vae import det_neg_elbo
from sequence.utils import anneal

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
    n_batches = n_total // batch_size

    c = 0
    for i in range(n_batches):
        global_step += 1
        c += 1
        optim.zero_grad()
        i = i * batch_size

        packed_padded, padded = dataset.get_batch(i, i + batch_size, device=device)
        nll, kl = det_neg_elbo(model, packed_padded, word_dropout)
        anneal_factor = anneal_f(global_step)
        loss = nll + kl * anneal_factor

        logger.debug(
            "{}/{}\t{}%\tLoss: {:.4f}".format(
                c, n_batches, int(c / n_batches * 100), loss.item()
            )
        )

        loss.backward()
        optim.step()

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalars(
                "Loss", {"NEG. ELBO": loss.item(), "NLL": nll.item(), "KL": kl.item()}
            )
            tensorboard_writer.add_scalar("Anneal factor", anneal_factor)

    logger.debug("Epoch: {}\tLoss{:.4f}".format(epoch, loss.item()))
