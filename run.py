import argparse
from sequence import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers()

    # GENERIC
    parser.add_argument("--logging_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--save_every_n", type=int, default=None, help="Save every n batches"
    )
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--storage_dir", type=str, default="storage")
    parser.add_argument("--tensorboard", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("--min_length", type=int, default=4, help="Minimum sequence length")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum sequence length")
    parser.add_argument("--train_percentage", type=float, default=0.9)
    parser.add_argument(
        "--dataset",
        type=str,
        default="brown",
        help="Pickled dataset file path, or named dataset (brown, treebank, Yoochoose 1/64). "
        "If none given, NLTK BROWN dataset will be used",
    )
    parser.add_argument("--force_cpu", type=bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--global_step", type=int, default=0, help="Overwrite global step."
    )
    parser.add_argument(
        "--continue",
        type=str,
        default=None,
        help="Path to existing ModelRegistry",
        dest="model_registry_path",
    )

    # VAE
    vae_parser = subparsers.add_parser(
        "vae", help="Run VAE model", parents=[parser], add_help=False
    )
    vae_parser.set_defaults(func=main.vae.main)
    vae_parser.add_argument("--hidden_size", type=int, default=64)
    vae_parser.add_argument("--latent_size", type=int, default=100)
    vae_parser.add_argument("-d", "--word_dropout", type=float, default=0.75)
    vae_parser.add_argument(
        "--annealing_epochs",
        type=float,
        default=3.0,
        help="In how many epochs the annealing should be 1.",
    )

    # STAMP
    stamp_parser = subparsers.add_parser(
        "stamp", help="Run ST(A)MP model", parents=[parser], add_help=False
    )
    stamp_parser.set_defaults(func=main.stamp.main)
    stamp_parser.add_argument("--nonlinearity", type=str, default="tanh")
    stamp_parser.add_argument("--model", type=str, default="stmp", help="stmp or stamp")

    args = parser.parse_args()
    args.func(args)
