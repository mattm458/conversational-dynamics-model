#! python
import json
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from dataset import ConversationDataset, collate_fn
from model.sequential_conversation import SequentialConversationModel

if __name__ == "__main__":
    from util.args import args

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # Load the main experiment configuration file
    with open(args.config, "r") as infile:
        config = json.load(infile)

    # Check the mode parameter from the arguments.
    # The mode will determine the behavior of the main program.
    if args.mode == "test":
        # Evaluate a trained model on a validation or test set, and output the results.
        pass

    elif args.mode == "train" or args.mode == "lr":
        # Train a new model from scratch, or resume training from a checkpoint.
        train_ses_ids = torch.load(args.train_ids)
        val_ses_ids = torch.load(args.val_ids)

        train_dataset = ConversationDataset(
            ses_ids=train_ses_ids,
            features_dir=args.dataset_dir,
            embeddings_dir=args.embeddings_dir,
            speech_feature_keys=config["speech_feature_keys"],
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
            **config["dataloader"],
        )

        val_dataset = ConversationDataset(
            ses_ids=val_ses_ids,
            features_dir=args.dataset_dir,
            embeddings_dir=args.embeddings_dir,
            speech_feature_keys=config["speech_feature_keys"],
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            **config["dataloader"],
        )

        model = SequentialConversationModel(
            speech_feature_keys=config["speech_feature_keys"],
            num_speech_features=len(config["speech_feature_keys"]),
            **config["model"],
        )

        if args.mode == "lr":
            trainer = pl.Trainer(auto_lr_find=True, **config["trainer"])

            trainer.tune(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
        elif args.mode == "train":
            callbacks = [
                ModelCheckpoint(
                    save_top_k=10,
                    monitor="val_loss",
                    mode="min",
                    filename="dialogue-{epoch:02d}-{val_loss:.5f}",
                )
            ]

            if config["trainer_plugins"]["early_stopping"]["active"]:
                callbacks.append(
                    EarlyStopping(
                        **config["trainer_plugins"]["early_stopping"]["params"]
                    )
                )

            trainer = pl.Trainer(
                callbacks=callbacks,
                devices=[args.device],
                **config["trainer"],
            )

            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

    elif args.mode == "torchscript":
        # Export the model to TorchScript
        filename = args.filename
        if filename is None:
            filename = f"neural-entrainment-{int(time.time())}.pt"

        # TODO: args.checkpoint should be required for Torchscript export!
        # Only optional for testing purposes during development
        if args.checkpoint is not None:
            model = SequentialConversationModel.load_from_checkpoint(
                args.checkpoint,
                num_speech_features=len(config["speech_feature_keys"]),
                **config["model"],
            )
        else:
            model = SequentialConversationModel(**config["model"])

        # Do the export
        model.to_torchscript(filename)
