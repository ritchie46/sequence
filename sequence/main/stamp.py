from sequence.model.stamp import STMP, STAMP
import os
from sequence.train.stamp import run_epoch
from sequence.main import generic
from sequence.callbacks import log_ranking_metrics


def main(args):

    name = args.logging_name
    if args.model == "stmp":
        cls = STMP
        if name is None:
            name = "stmp"
    else:
        if name is None:
            name = "stamp"
        cls = STAMP
    artifact_dir, tb_dir = generic.create_dirs(args, name)
    dataset_train, dataset_test, language = generic.load_dataset(args)
    model_args = dict(
        vocabulary_size=language.vocabulary_size,
        embedding_dim=args.embedding_dim,
        nonlinearity=args.nonlinearity,
    )

    model_registry = generic.load_model_registry(args, cls, name, **model_args)

    optim = generic.init_optimizer(args, model_registry)

    writer = generic.init_tensorboard(args, tb_dir, name)
    callbacks_ = generic.init_callbacks(args, model_registry, artifact_dir)
    global_step = generic.init_global_step(args, model_registry)
    device = generic.init_device(args, model_registry)
    callbacks_.append(log_ranking_metrics(50, k=20))

    for e in range(args.epochs):
        global_step = run_epoch(
            e,
            model_registry.model_,
            optim,
            dataset_train,
            dataset_test,
            args.batch_size,
            device=device,
            tensorboard_writer=writer,
            global_step=global_step,
            callbacks=callbacks_,
            scale_loss_by_lengths=args.scale_loss_by_lengths == "true",
        )

        with open(os.path.join(artifact_dir, f"{e}.pkl"), "wb") as f:
            model_registry.dump(f)
