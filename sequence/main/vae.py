from sequence.model.vae import VAE
import os
from sequence.utils import annealing_sigmoid
from sequence.train.vae import run_epoch
from sequence.main import generic


def main(args):

    dataset, language = generic.load_ds(args)
    model_registry = generic.load_model_registry(
        args,
        VAE,
        "vae",
        **dict(
            vocabulary_size=language.vocabulary_size,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            bidirectional=False,
            rnn_layers=1,
        ),
    )

    optim = generic.init_optimizer(args, model_registry)

    name = f"z{args.latent_size}-h{args.hidden_size}-e{args.embedding_dim}-wd{args.word_dropout}"
    artifact_dir, tb_dir = generic.create_dirs(args, name)
    writer = generic.init_tensorboard(args, tb_dir, name)
    callbacks_ = generic.init_callbacks(args, model_registry, artifact_dir)
    global_step = generic.init_global_step(args, model_registry)
    device = generic.init_device(args, model_registry)

    def anneal_f(i):
        pct = i / (len(dataset) * args.annealing_epochs) * args.batch_size
        return annealing_sigmoid(0, 1, pct)

    for e in range(args.epochs):
        global_step = run_epoch(
            e,
            model_registry.model_,
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
            model_registry.dump(f)
