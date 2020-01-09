import pickle
from sequence.model.vae import VAE
from sequence.data.datasets import treebank, brown
import os
from sequence.utils import annealing_sigmoid
from sequence import callbacks
from sequence.train.vae import run_epoch
import logging
from tensorboardX import SummaryWriter
import torch
from dumpster.registries.file import ModelRegistry


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    callbacks_ = []
    fn = args.dataset
    if fn == "treebank":
        dataset, language = treebank()
        fn = "NLTK " + fn
    elif fn == "brown":
        fn = "NLTK " + fn
        dataset, language = brown()
    else:
        with open(fn, "rb") as f:
            dataset = pickle.load(f)
            language = dataset.language

    logger.info(f"Using {fn} dataset")
    dataset, _ = dataset.split(
        [args.train_percentage, 1 - args.train_percentage], shuffle=False
    )

    logging.info(f"VOCABULARY SIZE: {language.vocabulary_size}")

    if args.model_registry_path is None:
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
    else:
        with open(args.model_registry_path, "rb") as f:
            mr = ModelRegistry().load(f)

    if torch.cuda.is_available() and not args.force_cpu:
        device = "cuda"
        logger.info("CUDA available.")
        mr.model_.cuda()
    else:
        logger.info("Running on CPU.")
        device = "cpu"

    optim = torch.optim.Adam(
        mr.model_.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    os.makedirs(args.storage_dir, exist_ok=True)

    name = f"z{args.latent_size}/h{args.hidden_size}/e{args.embedding_dim}/wd{args.word_dropout}"
    artifact_dir = os.path.join(args.storage_dir, "artifacts", name)
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
        callbacks_.append(
            callbacks.save_every_n_steps(
                n=args.save_every_n, mr=mr, dump_dir=artifact_dir
            )
        )
    callbacks_.append(callbacks.register_global_step(mr))

    global_step = args.global_step
    # Only use ModelRegistries global step if global_step is non default.
    if hasattr(mr, "global_step_") and global_step == 0:
        global_step = args.global_step

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
            callbacks=callbacks_,
        )

        with open(os.path.join(artifact_dir, f"{e}.pkl"), "wb") as f:
            mr.dump(f)