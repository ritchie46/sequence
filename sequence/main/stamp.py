from sequence.model.stamp import STMP, STAMP
import os
from sequence.train.stamp import run_epoch
from sequence.main import generic


def main(args):

    name = f"e{args.embedding_dim}"
    artifact_dir, tb_dir = generic.create_dirs(args, name)
    dataset, language = generic.load_dataset(args)
    model_args = dict(
        vocabulary_size=language.vocabulary_size,
        embedding_dim=args.embedding_dim,
        nonlinearity=args.nonlinearity,
    )
    if args.model == "stmp":
        cls = STMP
        name = args.logging_name if args.logging_name is not None else "stmp"
    else:
        cls = STAMP

    model_registry = generic.load_model_registry(
        args,
        cls,
        name,
        **model_args
    )

    optim = generic.init_optimizer(args, model_registry)

    writer = generic.init_tensorboard(args, tb_dir, name)
    callbacks_ = generic.init_callbacks(args, model_registry, artifact_dir)
    global_step = generic.init_global_step(args, model_registry)
    device = generic.init_device(args, model_registry)

    for e in range(args.epochs):
        global_step = run_epoch(
            e,
            model_registry.model_,
            optim,
            dataset,
            args.batch_size,
            device=device,
            tensorboard_writer=writer,
            global_step=global_step,
            callbacks=callbacks_,
            scale_loss_by_lengths=args.scale_loss_by_lengths == 'true'
        )

        with open(os.path.join(artifact_dir, f"{e}.pkl"), "wb") as f:
            model_registry.dump(f)
