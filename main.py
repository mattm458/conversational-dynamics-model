#! python
import json
import time

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from dataset import ConversationDataset, collate_fn
from model.sequential_conversation import SequentialConversationModel


FEATURES = [
    "pitch_mean_norm_clip",
    "pitch_range_norm_clip",
    "intensity_mean_norm_clip",
    "jitter_norm_clip",
    "shimmer_norm_clip",
    "nhr_norm_clip",
    "rate_norm_clip",
]

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

    elif args.mode == "train":
        # Train a new model from scratch, or resume training from a checkpoint.
        df = pd.read_csv(args.dataset)
        ses_ids = df.ses_id.unique()
        df = df.set_index("ses_id")

        train_ses, test_ses = train_test_split(
            ses_ids, train_size=0.8, random_state=9001
        )
        train_ses, val_ses = train_test_split(
            train_ses, train_size=0.8, random_state=9001
        )

        train_dataset = ConversationDataset(
            df,
            ses_ids=train_ses,
            embeddings_dir=args.embeddings_dir,
            speech_feature_keys=FEATURES,
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            collate_fn=collate_fn,
            persistent_workers=True,
            num_workers=16,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        val_dataset = ConversationDataset(
            df,
            ses_ids=val_ses,
            embeddings_dir=args.embeddings_dir,
            speech_feature_keys=FEATURES,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=32,
            collate_fn=collate_fn,
            persistent_workers=True,
            num_workers=16,
            pin_memory=True,
        )

        model = SequentialConversationModel(
            speech_feature_keys=FEATURES,
            teacher_forcing=0.5,
            lr=0.008,
        )

        trainer = pl.Trainer(
            devices=[0],
            precision=16,
            accelerator="gpu",
            callbacks=[
                ModelCheckpoint(
                    save_top_k=10,
                    monitor="val_loss",
                    mode="min",
                    filename="dialogue-{epoch:02d}-{val_loss:.2f}",
                ),
                EarlyStopping(monitor="val_loss", mode="min", patience=5),
            ],
        )

        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )
        # trainer = pl.Trainer(
        #     auto_lr_find=True, devices=[0], precision=16, accelerator="gpu"
        # )
        # trainer.tune(
        #     model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        # )

    elif args.mode == "torchscript":
        # Export the model to TorchScript
        filename = args.filename
        if filename is None:
            filename = f"neural-entrainment-{int(time.time())}.pt"

        # TODO: args.checkpoint should be required for Torchscript export!
        # Only optional for testing purposes during development
        if args.checkpoint is not None:
            model = SequentialConversationModel.load_from_checkpoint(
                args.checkpoint, **config["model"]
            )
        else:
            model = SequentialConversationModel(**config["model"])

        # Do the export
        model.to_torchscript(filename)
