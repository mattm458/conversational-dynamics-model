import matplotlib
import pytorch_lightning as pl

matplotlib.use("Agg")

from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import functional as F

from model.conversation import ConversationModel
from model.util import get_embeddings_subsequence, init_hidden, lengths_to_mask


class SequentialConversationModel(pl.LightningModule):
    def __init__(
        self,
        num_speech_features: int = 7,
        encoder_out_dim: int = 64,
        encoder_num_layers: int = 2,
        encoder_dropout: float = 0.5,
        attention_dim: int = 64,
        decoder_out_dim: int = 32,
        decoder_num_layers: int = 2,
        decoder_dropout: float = 0.5,
        decoder_attention_dropout: float = 0.0,
        num_decoders: int = 7,
        outputs_per_decoder: int = 1,
        embedding_dim: int = 300,
        embedding_encoder_out_dim: int = 64,
        embedding_encoder_num_layers: int = 2,
        embedding_encoder_dropout: float = 0.5,
        embedding_attention_dim: int = 64,
        embedding_attention_dropout: float = 0.0,
        has_speaker: bool = True,
        has_embeddings: bool = True,
        teacher_forcing: float = 0.5,
        speech_feature_keys: List[str] = [
            "pitch_mean_norm",
            "pitch_range_norm",
            "intensity_mean_norm",
            "jitter_norm",
            "shimmer_norm",
            "nhr_norm",
            "rate_norm",
        ],
        lr: float = 0.01,
        decoder_activation: str = "tanh",
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_speech_features: int = num_speech_features
        self.encoder_out_dim: int = encoder_out_dim
        self.encoder_num_layers: int = encoder_num_layers

        self.decoder_out_dim: int = decoder_out_dim
        self.decoder_num_layers: int = decoder_num_layers

        self.has_speaker: bool = has_speaker
        self.has_embeddings: bool = has_embeddings

        self.teacher_forcing = teacher_forcing

        self.speech_feature_keys = speech_feature_keys

        self.lr = lr

        self.conversation_model = ConversationModel(
            num_speech_features=num_speech_features,
            encoder_out_dim=encoder_out_dim,
            encoder_num_layers=encoder_num_layers,
            encoder_dropout=encoder_dropout,
            attention_dim=attention_dim,
            decoder_out_dim=decoder_out_dim,
            decoder_num_layers=decoder_num_layers,
            decoder_dropout=decoder_dropout,
            decoder_attention_dropout=decoder_attention_dropout,
            num_decoders=num_decoders,
            outputs_per_decoder=outputs_per_decoder,
            embedding_dim=embedding_dim,
            embedding_encoder_out_dim=embedding_encoder_out_dim,
            embedding_encoder_num_layers=embedding_encoder_num_layers,
            embedding_encoder_dropout=embedding_encoder_dropout,
            embedding_attention_dim=embedding_attention_dim,
            embedding_attention_dropout=embedding_attention_dropout,
            has_speaker=has_speaker,
            has_embeddings=has_embeddings,
            decoder_activation=decoder_activation,
        )

    def forward(
        self,
        speech_features: Tensor,
        encoder_hidden: List[Tuple[Tensor, Tensor]],
        decoder_hidden: List[Tuple[Tensor, Tensor]],
        history: Tensor,
        batch_mask: Tensor,
        encode_mask: Tensor,
        predict_mask: Tensor,
        dialogue_timestep: Tensor,
        speaker: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
        embeddings_len: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Pass through all arguments to the wrapped conversation model
        return self.conversation_model(
            speech_features=speech_features,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            history=history,
            batch_mask=batch_mask,
            encode_mask=encode_mask,
            predict_mask=predict_mask,
            dialogue_timestep=dialogue_timestep,
            speaker=speaker,
            embeddings=embeddings,
            embeddings_len=embeddings_len,
        )

    def predict_step(self, batch, batch_idx):
        X, y = batch

        outputs, attention_scores, attention_scores_len = self.sequence(
            speech_features=X["speech_features"],
            speech_features_len=X["speech_features_len"],
            dialogue_timestep=X["dialogue_timestep"],
            encode_mask=X["feature_encode_mask"],
            speaker=X["speaker"],
            embeddings=X["embeddings"],
            embeddings_subsequence_start=X["embeddings_idx_start"],
            embeddings_subsequence_end=X["embeddings_idx_end"],
            predict_mask=X["predict_mask"],
            autoregress_mask=X["autoregress_mask"],
            teacher_forcing=0.0,
            output_len=y["speech_features_len"].max(),
            us_count=y["us_count"],
        )

        return outputs, attention_scores, attention_scores_len

    def validation_step(self, batch, batch_idx):
        X, y = batch

        outputs, attention_scores, attention_scores_len = self.sequence(
            speech_features=X["speech_features"],
            speech_features_len=X["speech_features_len"],
            dialogue_timestep=X["dialogue_timestep"],
            encode_mask=X["feature_encode_mask"],
            speaker=X["speaker"],
            embeddings=X["embeddings"],
            embeddings_subsequence_start=X["embeddings_idx_start"],
            embeddings_subsequence_end=X["embeddings_idx_end"],
            predict_mask=X["predict_mask"],
            autoregress_mask=X["autoregress_mask"],
            teacher_forcing=0.0,
            output_len=y["speech_features_len"].max(),
            us_count=y["us_count"],
        )

        y_mask = ~lengths_to_mask(
            y["speech_features_len"], y["speech_features_len"].max()
        ).squeeze(-1)

        loss = F.smooth_l1_loss(outputs[y_mask], y["speech_features"][y_mask])
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        for i, key in enumerate(self.speech_feature_keys):
            feature_loss = F.smooth_l1_loss(
                outputs[:, :, i][y_mask], y["speech_features"][:, :, i][y_mask]
            )
            self.log(
                f"val_loss_{key}",
                feature_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        return {
            "loss": loss,
            "speech_features_len": y["speech_features_len"],
            "attention_scores": attention_scores,
            "attention_scores_len": attention_scores_len,
        }

    def training_step(self, batch, batch_idx):
        X, y = batch

        outputs, _, _ = self.sequence(
            speech_features=X["speech_features"],
            speech_features_len=X["speech_features_len"],
            dialogue_timestep=X["dialogue_timestep"],
            encode_mask=X["feature_encode_mask"],
            speaker=X["speaker"],
            embeddings=X["embeddings"],
            embeddings_subsequence_start=X["embeddings_idx_start"],
            embeddings_subsequence_end=X["embeddings_idx_end"],
            predict_mask=X["predict_mask"],
            autoregress_mask=X["autoregress_mask"],
            teacher_forcing=self.teacher_forcing,
            output_len=y["speech_features_len"].max(),
            us_count=y["us_count"],
        )

        y_mask = ~lengths_to_mask(
            y["speech_features_len"], y["speech_features_len"].max()
        ).squeeze(-1)
        loss = F.mse_loss(outputs[y_mask], y["speech_features"][y_mask])

        self.log(
            "train_loss",
            loss.detach(),
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def sequence(
        self,
        speech_features: Tensor,
        speech_features_len: Tensor,
        dialogue_timestep: Tensor,
        encode_mask: Tensor,
        predict_mask: Tensor,
        autoregress_mask: Tensor,
        output_len: int,
        us_count: Tensor,
        teacher_forcing: float,
        speaker: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
        embeddings_subsequence_start: Optional[Tensor] = None,
        embeddings_subsequence_end: Optional[Tensor] = None,
    ):
        # This method implements a sequential forward pass for training.
        # Its purpose is to manage the process of moving through the
        # dialogue, handle teacher forcing, and maintain hidden states
        # and the history tensor. It should never be used for inference
        # in a live setting - for this, use the standard forward() method.

        # Retrieve some basic information from the input
        batch_size: int = speech_features.shape[0]
        num_input_steps: int = speech_features.shape[1]
        num_dialogue_timesteps: int = int(dialogue_timestep.max())
        device: device = speech_features.device

        # Set up initial hidden states for the encoder and decoder
        encoder_hidden: List[Tuple[Tensor, Tensor]] = init_hidden(
            batch_size=batch_size,
            hidden_dim=self.encoder_out_dim,
            num_layers=self.encoder_num_layers,
            device=device,
        )
        decoder_hidden: List[List[Tuple[Tensor, Tensor]]] = [
            init_hidden(
                batch_size=batch_size,
                hidden_dim=self.decoder_out_dim,
                num_layers=self.decoder_num_layers,
                device=device,
            )
            for _ in range(7)
        ]

        # Set up a blank history tensor to accumulate encoded features
        history: Tensor = torch.zeros(
            (batch_size, num_dialogue_timesteps + 1, self.encoder_out_dim),
            device=device,
        )

        # Contains the previous output of the model, used for autoregressive training
        previous_output: Optional[Tensor] = None

        # Contains a mask showing which conversations in the batch had output at the
        # previous timestep. This allows us to determine if we need to encode anything
        # in `previous_output`.
        previous_predict_mask: Tensor = predict_mask[:, 0]

        # A tensor that collects all outputs from the model across all timesteps
        all_outputs: Tensor = torch.zeros(
            (batch_size, output_len, self.num_speech_features),
            device=device,
        )
        # Contains the index at which new outputs should be saved in `all_outputs`.
        # Not all conversations in the batch will save to the output tensor at once,
        # so this allows us to manage it on a conversation-by-conversation basis
        outputs_idx: Tensor = torch.zeros(batch_size, dtype=torch.long, device=device)

        # A tensor that collects all attention scores from the model across all timesteps
        all_attention_scores: Tensor = torch.zeros(
            (
                batch_size,
                max(us_count),
                num_dialogue_timesteps,
                self.num_speech_features,
            ),
            device=device,
        )

        all_attention_score_len = torch.zeros(
            (batch_size, max(us_count)), device=device, dtype=torch.long
        )

        embeddings_subsequence_start = torch.split(
            embeddings_subsequence_start, 1, dim=1
        )
        embeddings_subsequence_end = torch.split(embeddings_subsequence_end, 1, dim=1)
        predict_mask = torch.split(predict_mask, 1, dim=1)
        dialogue_timestep = torch.split(dialogue_timestep, 1, dim=1)
        speech_features = torch.split(speech_features, 1, dim=1)
        autoregress_mask = torch.split(autoregress_mask, 1, dim=1)
        speaker = torch.split(speaker, 1, dim=1)
        encode_mask = torch.split(encode_mask, 1, dim=1)

        # Iterate over all the input steps
        for timestep in range(num_input_steps):
            # Get the embeddings subsequence for the current timestep
            (
                embeddings_subsequence,
                embeddings_subsequence_len,
            ) = get_embeddings_subsequence(
                embeddings,
                embeddings_subsequence_start[timestep].squeeze(1),
                embeddings_subsequence_end[timestep].squeeze(1),
            )

            # Create the timestep batch mask. This mask contains True if the
            # current timestep contains active dialogue, and False if the
            # current timestep has gone beyond the end of the dialogue.
            timestep_batch_mask: Tensor = (
                torch.full((batch_size,), timestep, device=device) < speech_features_len
            )
            timestep_predict_mask: Tensor = predict_mask[timestep].squeeze(1)
            timestep_dialogue_timestep: Tensor = dialogue_timestep[timestep].squeeze(1)

            # Get the input speech features for the current timestep
            input_speech_features = speech_features[timestep].squeeze(1).clone()

            # Do we need to autoregress from the previous timestep?
            timestep_autoregress_mask: Tensor = autoregress_mask[timestep].squeeze(1)
            if (
                previous_output is not None
                and timestep_autoregress_mask.any()
                and previous_predict_mask.any()
            ):
                # If so, construct a random mask for teacher forcing.
                # Values above the threshhold mean we will keep the ground-truth
                # input speech features. Values below the threshhold mean we will
                # replace the ground-truth input speech features with values
                # predicted at the previous timestep.
                teacher_forcing_mask = (
                    torch.rand(timestep_autoregress_mask.shape, device=device)
                    < teacher_forcing
                )

                # Construct the final autoregression mask. Values of True mean
                # we will replace the ground-truth feature value (i.e., we
                # autoregress), while values of False mean we will keep the
                # ground-truth feature value (i.e., we teacher force)
                timestep_autoregress_mask = (
                    timestep_autoregress_mask * ~teacher_forcing_mask
                )

                # Apply the mask by assigning previously predicted speech features
                input_speech_features[timestep_autoregress_mask] = previous_output[
                    timestep_autoregress_mask[previous_predict_mask]
                ].type(input_speech_features.dtype)

            # Perform one step through the model
            model_output, attention_scores, history_seq_len = self(
                speech_features=input_speech_features,
                speaker=speaker[timestep].squeeze(1),
                encoder_hidden=encoder_hidden,
                decoder_hidden=decoder_hidden,
                history=history,
                batch_mask=timestep_batch_mask,
                encode_mask=encode_mask[timestep].squeeze(1),
                predict_mask=timestep_predict_mask,
                dialogue_timestep=timestep_dialogue_timestep,
                embeddings=embeddings_subsequence,
                embeddings_len=embeddings_subsequence_len,
            )

            # Save the model output and attention scores
            if (
                model_output is not None
                and attention_scores is not None
                and history_seq_len is not None
            ):
                # Get indices in the output tensor for saving
                timestep_outputs_idx = outputs_idx[timestep_predict_mask]

                # Save the output
                all_outputs[
                    timestep_predict_mask, timestep_outputs_idx
                ] = model_output.type(all_outputs.dtype)

                # Pad the attention scores so they are long enough to fit in the tensor
                attention_scores = F.pad(
                    attention_scores,
                    (0, 0, 0, num_dialogue_timesteps - attention_scores.shape[1]),
                )

                # Save the attention scores
                all_attention_scores[timestep_predict_mask, timestep_outputs_idx] = (
                    attention_scores.detach().clone().type(all_attention_scores.dtype)
                )

                all_attention_score_len[
                    timestep_predict_mask, timestep_outputs_idx
                ] = history_seq_len

                # Increment the indices we used by 1
                outputs_idx[timestep_predict_mask] += 1

                previous_output = model_output.detach().clone()
            else:
                previous_output = None

            previous_predict_mask = predict_mask[timestep].squeeze(1)

        return all_outputs, all_attention_scores, all_attention_score_len

    def validation_epoch_end(self, outputs):
        output = outputs[0]

        for i, feature in enumerate(self.speech_feature_keys):
            self.logger.experiment.add_image(
                f"val_alignment_{feature}",
                plot_feature_attention(
                    feature_idx=i,
                    attention=output["attention_scores"].cpu(),
                    batch_idx=0,
                    speech_features_len=output["speech_features_len"][0].cpu(),
                    attention_len=output["attention_scores_len"][0].cpu(),
                ),
                self.current_epoch,
                dataformats="HWC",
            )


def save_figure_to_numpy(fig):
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    return data


def plot_feature_attention(
    feature_idx: int,
    attention: Tensor,
    batch_idx: int,
    speech_features_len: Tensor,
    attention_len: Tensor,
) -> np.ndarray:
    ATT = attention[batch_idx, : speech_features_len.max(), :, feature_idx]
    rows, columns = ATT.shape
    fig, axs = plt.subplots(ncols=1, nrows=rows, figsize=(10, 10))

    for i, (scores, length) in enumerate(zip(ATT, attention_len)):
        ax = axs[i]
        ax.imshow(
            scores[:length].unsqueeze(0),
            interpolation="nearest",
            aspect="auto",
            vmin=0 if i == 0 else None,
            vmax=1 if i == 0 else None,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    return data
