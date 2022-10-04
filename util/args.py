import argparse


def train_subparser(subparsers):
    lr_subparser = subparsers.add_parser(
        "lr", help="Find an appropriate learning rate for a model configuration"
    )

    lr_subparser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to a directory with dialogue dataset features",
    )

    lr_subparser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Path to a directory with word embedings extracted from the dataset",
    )

    train_subparser = subparsers.add_parser(
        "train", help="Train a neural entrainment model"
    )

    train_subparser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Resume training from the given checkpoint.",
        default=None,
    )

    train_subparser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to a directory with dialogue dataset features",
    )

    train_subparser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Path to a directory with word embedings extracted from the dataset",
    )

    train_subparser.add_argument(
        "--device",
        type=int,
        required=True,
        help="The device to use for training",
    )

    train_subparser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="The number of dataloader workers",
    )


def test_subparser(subparsers):
    test_subparser = subparsers.add_parser(
        "test", help="Evaluate a trained neural entrainment model"
    )

    test_subparser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="A checkpoint from a trained model to evaluate.",
    )

    test_subparser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="The base dataset directory",
    )

    test_subparser.add_argument(
        "--out",
        type=str,
        required=True,
        help="A filename to save results output.",
    )


def torchscript_subparser(subparsers):
    torchscript_subparser = subparsers.add_parser(
        "torchscript", help="Export the model to TorchScript"
    )

    # TODO: The checkpoint argument should be required!
    # Only optional for testing purposes during development
    torchscript_subparser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="A trained checkpoint to export with the model.",
    )

    torchscript_subparser.add_argument(
        "--filename",
        type=str,
        required=False,
        help="The TorchScript model filename",
        default=None,
    )


parser = argparse.ArgumentParser(description="Train a neural entrainment engine")

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="A JSON configuration file containing model configurations and hyperparameters.",
)

subparsers = parser.add_subparsers(required=True, dest="mode")

train_subparser(subparsers)
test_subparser(subparsers)
torchscript_subparser(subparsers)

args = parser.parse_args()
