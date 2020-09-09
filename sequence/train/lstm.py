from sequence.model.lstm import det_loss
from sequence.utils import backward
from sequence.callbacks import apply as apply_callbacks
import logging

logger = logging.getLogger(__name__)


def run_epoch(
    epoch,
    model,
    optim,
    dataset_train,
    dataset_test,
    batch_size,
    device="cpu",
    tensorboard_writer=None,
    global_step=0,
    n_batches=None,
    callbacks=[],
    scale_loss_by_lengths=True,
):
    model.train()
    n_total = len(dataset_train)
    if n_batches is None:
        n_batches = n_total // batch_size

    state_h, state_c = model.init_state(batch=batch_size)

    c = 0
    for i in range(n_batches):
        epoch_p = i / (n_batches - 1)
        global_step += 1
        c += 1

        optim.zero_grad()
        i = i * batch_size

        packed_padded, padded = dataset_train.get_batch(
            i, i + batch_size, device=device
        )

        loss = det_loss(
            model,
            packed_padded,
            scale_loss_by_lengths=scale_loss_by_lengths,
            max_len=dataset_train.max_len,
            state_h=state_h,
            state_c=state_c,
        )
        state_h = state_h.detach()
        state_c = state_c.detach()

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

        apply_callbacks(
            callbacks,
            global_step=global_step,
            loss=loss,
            model=model,
            ds_train=dataset_train,
            ds_test=dataset_test,
            logger=logger,
            device=device,
            epoch_p=epoch_p,
            tensorboard_writer=tensorboard_writer,
        )

    logger.debug("Epoch: {}\tLoss{:.4f}".format(epoch, loss.item()))
    return global_step
