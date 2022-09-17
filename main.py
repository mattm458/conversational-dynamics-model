#! python
import json
import time

from model.neural_entrainment import NeuralEntrainment
from util.args import args

if __name__ == "__main__":
    # Load the main experiment configuration file
    with open(args.config, "r") as infile:
        config = json.load(infile)

    # Check the mode parameter from the arguments.
    # The mode will determine the behavior of the main program.
    if args.mode == "train":
        # Train a new model from scratch, or resume training from a checkpoint.
        pass

    elif args.mode == "test":
        # Evaluate a trained model on a validation or test set, and output the results.
        pass

    elif args.mode == "torchscript":
        # Export the model to TorchScript
        filename = args.filename
        if filename is None:
            filename = f"neural-entrainment-{int(time.time())}.pt"

        # TODO: args.checkpoint should be required for Torchscript export!
        # Only optional for testing purposes during development
        if args.checkpoint is not None:
            model = NeuralEntrainment.load_from_checkpoint(
                args.checkpoint, **config["model"]
            )
        else:
            model = NeuralEntrainment(**config["model"])

        # Do the export
        model.to_torchscript(filename)
