from clickstream.seq2seq import decoder_loss, decoder_loss_batched
import logging

logger = logging.getLogger(__name__)


def run_epoch(
    e,
    m,
    optim,
    dataset,
    batch_size,
    teach_forcing_p=0.5,
    device="cpu",
    nullify_rnn_input=False,
    slow=False,
    verbosity=0,
):
    m.train()
    n_total = len(dataset)
    dataset.shuffle()

    for i in range(n_total // batch_size):
        optim.zero_grad()
        i = i * batch_size

        packed_padded, padded = dataset.get_batch(i, i + batch_size, device=device)
        if slow:
            loss = decoder_loss(m, padded)
        else:
            z = m.encode(packed_padded)
            loss = decoder_loss_batched(
                m, z, packed_padded, teach_forcing_p, nullify_rnn_input
            )
        loss.backward()
        optim.step()

    level = logging.DEBUG if verbosity == 0 else logging.INFO
    logger.log(level, "Epoch: {}\tLoss{:.4f}".format(e, loss.item()))
