import pickle

from sequence.model.vae import VAE
from sequence.data.datasets import treebank
import os
from sequence.utils import annealing_sigmoid
from sequence import callbacks
from sequence.train.vae import run_epoch
import logging
from tensorboardX import SummaryWriter
import torch
from dumpster.registries.file import ModelRegistry
import argparse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    fn = args.dataset
    if fn is None:
        logger.info("Using NLTK treebank dataset")
        dataset, language = treebank()
    else:
        logger.info(f"Using dataset from {fn}")
        with open(fn, "rb") as f:
            dataset = pickle.load(f)
    dataset, _ = dataset.split(
        [args.train_percentage, 1 - args.train_percentage], shuffle=False
    )

    logging.info(f"VOCABULARY SIZE: {language.vocabulary_size}")

    mr = ModelRegistry("vae")
    mr.register(
        VAE,
        insert_methods="pytorch",
        **dict(
            vocabulary_size=language.vocabulary_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            bidirectional=False,
            rnn_layers=1,
        ),
    )

    if torch.cuda.is_available() and not args.force_cpu:
        device = "cuda"
        logger.info("CUDA available.")
        mr.model_.cuda()
    else:
        logger.info("Running on CPU.")
        device = "cpu"

    optim = torch.optim.Adam(mr.model_.parameters(), lr=args.lr)
    os.makedirs(args.storage_dir, exist_ok=True)
    artifact_dir = os.path.join(args.storage_dir, "artifacts")

    name = f"z{args.latent_size}/h{args.hidden_size}/e{args.embedding_dim}/wd{args.word_dropout}"
    if args.tensorboard:
        tb_dir = os.path.join(args.storage_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(tb_dir, name))
    else:
        writer = None

    def anneal_f(i):
        pct = i / (len(dataset) * args.annealing_epochs) * args.batch_size
        return annealing_sigmoid(0, 1, pct)

    os.makedirs(os.path.join(artifact_dir, name), exist_ok=True)
    if args.save_every_n is not None:
        callbacks.save_every_n_steps(n=args.save_every_n, mr=mr, dump_dir=artifact_dir)

    global_step = 0
    for e in range(args.epochs):
        global_step = run_epoch(
            e,
            mr.model_,
            optim,
            dataset,
            args.batch_size,
            word_dropout=args.word_dropout,
            device=device,
            tensorboard_writer=writer,
            global_step=global_step,
            anneal_f=anneal_f,
        )

        with open(os.path.join(artifact_dir, name + f"-{e}.pkl"), "wb") as f:
            mr.dump(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--latent_size", type=int, default=100)
    parser.add_argument("-d", "--word_dropout", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--storage_dir", type=str, default="storage")
    parser.add_argument("--tensorboard", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("--min_length", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--train_percentage", type=float, default=0.9)
    parser.add_argument(
        "--annealing_epochs",
        type=float,
        default=3.0,
        help="In how many epochs the annealing should be 1.",
    )
    parser.add_argument(
        "--save_every_n", type=int, default=None, help="Save every n batches"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Pickled dataset file. If none given, NLTK BROWN dataset will be used",
    )
    parser.add_argument("--force_cpu", type=bool, default=False)

    args = parser.parse_args()
    main(args)
