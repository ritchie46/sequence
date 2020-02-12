import torch
from functools import partial
import numpy as np
from sequence.model.seq2seq import run_decoder, EncoderDecoder
from sequence.model.vae import VAE
from sequence.model.stamp import STMP
from sequence.train.ae import run_epoch
from sequence.utils import anneal
from sequence.test.test_ae import dataset, language, paths, words

try:
    from apex import amp

    opt_level = "00"  # no mixed precision.
except ImportError:
    pass


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
    if globals().get("amp", False):
        m, optim = amp.initialize(m, optim, opt_level=opt_level)
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
    if globals().get("amp", False):
        m, optim = amp.initialize(m, optim, opt_level=opt_level)
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


def test_vae(dataset, language):
    from sequence.train.vae import run_epoch
    from sequence.model.vae import run_decoder

    torch.manual_seed(0)
    np.random.seed(0)

    latent_size = 32
    batch_size = 64
    word_dropout = 1.0
    m = VAE(
        vocabulary_size=language.vocabulary_size,
        embedding_dim=8,
        hidden_size=16,
        latent_size=latent_size,
        bidirectional=False,
    )
    device = "cuda"
    if device == "cuda":
        m.cuda()
    optim = torch.optim.Adam(m.parameters(), lr=0.001)
    if globals().get("amp", False):
        m, optim = amp.initialize(m, optim, opt_level=opt_level)
    anneal_f = partial(anneal, goal=1000 / batch_size, f="other")

    for e in range(50):
        run_epoch(
            e,
            m,
            optim,
            dataset,
            batch_size,
            word_dropout=word_dropout,
            device=device,
            anneal_f=anneal_f,
        )

    packed_padded, padded = dataset.get_batch(0, 25, device=device)
    h, z, mu, log_var = m.encode(packed_padded)

    out, target = run_decoder(m, packed_padded, word_dropout, h)
    out = out.argmax(-1)

    for i in range(25):
        t = target[i, :]
        mask = t >= 0
        print(t[mask], out[i, :][mask])


def test_stmp(dataset, language):
    torch.manual_seed(0)
    np.random.seed(0)
    m = STMP(
        language.vocabulary_size, embedding_dim=8, mlp_layers=1, nonlinearity="tanh"
    )

    device = "cuda"
    if device == "cuda":
        m.cuda()
    optim = torch.optim.Adam(m.parameters(), lr=0.01)
    from sequence.train.stamp import run_epoch

    run_epoch(2, m, optim, dataset, batch_size=32, device=device, dataset_test=dataset)

    packed_padded, padded = dataset.get_batch(0, 200, device=device)
    y_hat = m(packed_padded)

    from sequence.metrics import rank_scores

    print(rank_scores(y_hat.cpu(), padded.T[:, 1:].cpu(), k=3))
