from sequence.data import datasets
import pickle
import logging
import os
from tensorboardX import SummaryWriter
from sequence import callbacks
from dumpster.registries.file import ModelRegistry
import torch
import argparse
from torch import nn
from typing import List, Callable


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(args: argparse.Namespace):
    dataset_kwargs = dict(min_len=args.min_length, max_len=args.max_length)
    fn = args.dataset
    if fn == "treebank":
        dataset, language = datasets.treebank(dataset_kwargs)
        fn = "NLTK " + fn
    elif fn == "brown":
        fn = "NLTK " + fn
        dataset, language = datasets.brown(dataset_kwargs)
    elif fn == "Yoochoose 1/64":
        dataset_kwargs["min_len"] = 1
        dataset_kwargs["max_len"] = None
        dataset, language = datasets.yoochoose(
            args.storage_dir, div64=True, dataset_kwargs=dataset_kwargs
        )
    else:
        with open(fn, "rb") as f:
            dataset = pickle.load(f)
            language = dataset.language

    logger.info(f"Using {fn} dataset")
    dataset_train, dataset_test = dataset.split(
        [args.train_percentage, 1 - args.train_percentage], shuffle=False
    )
    logging.info(f"VOCABULARY SIZE: {language.vocabulary_size}")
    return dataset_train, dataset_test, language


def load_model_registry(
    args: argparse.Namespace, cls: nn.Module, name: str, **kwargs: dict
) -> ModelRegistry:
    if args.model_registry_path is None:
        model_registry = ModelRegistry(name)
        model_registry.register(cls, insert_methods="pytorch", **kwargs)
    else:
        with open(args.model_registry_path, "rb") as f:
            model_registry = ModelRegistry().load(f)
    return model_registry


def create_dirs(args: argparse.Namespace, name: str) -> (str, str):
    os.makedirs(args.storage_dir, exist_ok=True)
    artifact_dir = os.path.join(args.storage_dir, "artifacts", name)
    os.makedirs(os.path.join(artifact_dir), exist_ok=True)
    tb_dir = os.path.join(args.storage_dir, "tb")
    if args.tensorboard:
        os.makedirs(tb_dir, exist_ok=True)

    return artifact_dir, tb_dir


def init_tensorboard(args: argparse.Namespace, tb_dir: str, name: str) -> SummaryWriter:
    if args.tensorboard:
        writer = SummaryWriter(os.path.join(tb_dir, name))
    else:
        writer = None
    return writer


def init_callbacks(
    args: argparse.Namespace, model_registry: ModelRegistry, artifact_dir: str
) -> List[Callable]:
    callbacks_ = []
    if args.save_every_n is not None:
        callbacks_.append(
            callbacks.save_every_n_steps(
                n=args.save_every_n, mr=model_registry, dump_dir=artifact_dir
            )
        )
    callbacks_.append(callbacks.register_global_step(model_registry))
    return callbacks_


def init_global_step(args: argparse.Namespace, model_registry: ModelRegistry) -> int:
    global_step = args.global_step
    # Only use ModelRegistries global step if global_step is non default.
    if hasattr(model_registry, "global_step_") and global_step == 0:
        global_step = args.global_step
    return global_step


def init_device(args: argparse.Namespace, model_registry: ModelRegistry) -> str:
    if torch.cuda.is_available() and not args.force_cpu:
        device = "cuda"
        logger.info("CUDA available.")
        model_registry.model_.cuda()
    else:
        logger.info("Running on CPU.")
        device = "cpu"
    return device


def init_optimizer(args: argparse.Namespace, model_registry: ModelRegistry):
    if args.optimizer == "adam":
        logger.info("Using adam optimizer")
        return torch.optim.Adam(
            model_registry.model_.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        logger.info("Using sgd optimizer")
        return torch.optim.SGD(
            model_registry.model_.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
