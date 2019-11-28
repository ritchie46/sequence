import torch
import numpy as np
from clickstream.model.seq2seq import run_decoder, EncoderDecoder
from clickstream.train.ae import run_epoch
from clickstream.test.test_ import dataset, language, paths, words


def test_non_batched(dataset, language):
    torch.manual_seed(0)
    np.random.seed(0)

    latent_size = 32
    batch_size = 64
    m = EncoderDecoder(
        vocabulary_size=language.vocabulary_size,
        embedding_dim=8,
        latent_size=latent_size,
        bidirectional=True,
        rnn_layers=1,
    )
    device = "cuda"
    if device == "cuda":
        m.cuda()
    optim = torch.optim.Adam(m.parameters(), lr=0.01)
    for _ in range(10):
        run_epoch(1, m, optim, dataset, batch_size, device=device, batched=False)

    packed_padded, padded = dataset.get_batch(0, 25, device=device)

    for i in range(padded.shape[1]):
        target = padded[:, i]
        target = target[target >= 0]

        z = m.encode(target)
        pred = []
        for j in range(len(target)):
            out, z = m.decode(word=None, h=z)
            pred.append(out.argmax(-1).item())
        print(target, pred)


def test_batched(dataset, language):
    torch.manual_seed(0)
    np.random.seed(0)

    latent_size = 16
    batch_size = 64
    m = EncoderDecoder(
        vocabulary_size=language.vocabulary_size,
        embedding_dim=8,
        latent_size=latent_size,
        bidirectional=True,
        rnn_layers=1,
    )

    device = "cuda"
    if device == "cuda":
        m.cuda()
    optim = torch.optim.Adam(m.parameters(), lr=0.01)
    for e in range(10):
        run_epoch(
            e,
            m,
            optim,
            dataset,
            batch_size,
            device=device,
            nullify_rnn_input=True,
            reverse_target=True,
        )

    packed_padded, padded = dataset.get_batch(0, 25, device=device)
    z = m.encode(packed_padded)
    out = run_decoder(m, z, padded, nullify_rnn_input=True)
    # invert padding
    _, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_padded)

    for i in range(padded.shape[1]):
        print(padded[: lengths[i], i].cpu(), out[i, : lengths[i]].cpu())
