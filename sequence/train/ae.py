import logging
from sequence.model.seq2seq import det_loss, det_loss_batched
from sequence.utils import backward
from sequence.callbacks import apply as apply_callbacks


logger = logging.getLogger(__name__)


def run_epoch(
    epoch,
    model,
    optim,
    dataset,
    batch_size,
    teach_forcing_p=0.5,
    device="cpu",
    nullify_rnn_input=False,
    batched=True,
    reverse_target=False,
    tensorboard_writer=None,
    global_step=0,
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
    teach_forcing_p : float
    device : str
        "cpu" or "cuda"
    nullify_rnn_input : bool
    batched : bool
    reverse_target : bool
        Predict the inverted sentence. Leads to cleaner representations.
    tensorboard_writer : SummaryWriter
    global_step : int
        Step counter
    callbacks : list[function[**kwargs]]
    """
    model.train()
    n_total = len(dataset)
    dataset.shuffle()
    n_batches = n_total // batch_size

    c = 0
    for i in range(n_batches):
        epoch_p = i / (n_batches - 1)
        c += 1
        optim.zero_grad()
        i = i * batch_size

        packed_padded, padded = dataset.get_batch(i, i + batch_size, device=device)
        if not batched:
            if not nullify_rnn_input:
                logger.warning(
                    "Argmument `nullify_rnn_input` is not used when using non batched learning."
                )
            logger.warning(
                "Argument `teacher_forcing_p` is not used when using non batched learning."
            )
            loss = det_loss(model, padded)
        else:
            loss = det_loss_batched(
                model, packed_padded, teach_forcing_p, nullify_rnn_input, reverse_target
            )
            logger.debug(
                "{}/{}\t{}%\tLoss: {:.4f}".format(
                    c, n_batches, int(c / n_batches * 100), loss.item()
                )
            )
        backward(loss, optim)
        optim.step()

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("Loss", loss.item())

        apply_callbacks(
            callbacks,
            global_step=global_step,
            loss=loss,
            model=model,
            ds_train=dataset,
            logger=logger,
            device=device,
            epoch_p=epoch_p,
            tensorboard_writer=tensorboard_writer,
        )

    logger.debug("Epoch: {}\tLoss{:.4f}".format(epoch, loss.item()))
