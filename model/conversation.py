from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from model.components import Decoder, Encoder
from model.util import get_hidden_vector
from model.words import EmbeddingEncoder


class ConversationModel(nn.Module):
    def __init__(
        self,
        num_speech_features: int,
        encoder_out_dim: int,
        encoder_num_layers: int,
        encoder_dropout: float,
        attention_dim: int,
        decoder_out_dim: int,
        decoder_num_layers: int,
        decoder_dropout: float,
        decoder_attention_dropout: float,
        num_decoders: int,
        outputs_per_decoder: int,
        embedding_dim: int,
        embedding_encoder_out_dim: int,
        embedding_encoder_num_layers: int,
        embedding_encoder_dropout: float,
        embedding_attention_dim: int,
        embedding_attention_dropout: float,
        has_speaker: bool,
        has_embeddings: bool,
    ):
        super().__init__()

        self.has_speaker: bool = has_speaker
        self.has_embeddings: bool = has_embeddings

        encoder_in_dim = num_speech_features
        if has_speaker:
            encoder_in_dim += 2
        if has_embeddings:
            encoder_in_dim += embedding_encoder_out_dim

        self.embedding_encoder: Optional[EmbeddingEncoder] = None

        if has_embeddings:
            self.embedding_encoder = EmbeddingEncoder(
                embedding_dim=embedding_dim,
                encoder_out_dim=embedding_encoder_out_dim,
                encoder_num_layers=embedding_encoder_num_layers,
                encoder_dropout=embedding_encoder_dropout,
                attention_dim=embedding_attention_dim,
                attention_dropout=embedding_attention_dropout,
            )

        self.encoder: Encoder = Encoder(
            in_dim=encoder_in_dim,
            hidden_dim=encoder_out_dim,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
        )

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    attention_in_dim=encoder_out_dim,
                    attention_context_dim=decoder_out_dim + embedding_encoder_out_dim,
                    attention_dim=attention_dim,
                    decoder_in_dim=encoder_out_dim + embedding_encoder_out_dim,
                    decoder_dropout=decoder_dropout,
                    hidden_dim=decoder_out_dim,
                    output_dim=outputs_per_decoder,
                    num_layers=decoder_num_layers,
                    attention_dropout=decoder_attention_dropout,
                )
                for _ in range(num_decoders)
            ]
        )

    def forward(
        self,
        speech_features: Tensor,
        encoder_hidden: List[Tuple[Tensor, Tensor]],
        decoder_hidden: List[List[Tuple[Tensor, Tensor]]],
        history: Tensor,
        batch_mask: Tensor,
        encode_mask: Tensor,
        predict_mask: Tensor,
        dialogue_timestep: Tensor,
        speaker: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
        embeddings_len: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:

        # If we are encoding embeddings, do it now
        embeddings_encoded: Optional[Tensor] = None
        if (
            self.embedding_encoder is not None
            and embeddings is not None
            and embeddings_len is not None
        ):
            embeddings_encoded, _ = self.embedding_encoder(
                encoder_in=embeddings[batch_mask],
                lengths=embeddings_len[batch_mask],
            )

        # If there are any features to encode, encode them here
        if encode_mask.any():
            encoder_in: List[Tensor] = [speech_features[encode_mask]]

            # Add speaker indicator if required by the model
            if self.has_speaker and speaker is not None:
                encoder_in.append(speaker[encode_mask])

            # Add embedding vector if required by the model
            if embeddings_encoded is not None:
                embeddings_encode_mask = encode_mask[batch_mask]
                encoder_in.append(embeddings_encoded[embeddings_encode_mask])

            encoder_in: Tensor = torch.cat(encoder_in, dim=1)
            encoded = self.encoder(
                encoder_input=encoder_in, hidden=encoder_hidden, mask=encode_mask
            )

            history[encode_mask, dialogue_timestep[encode_mask]] = encoded.type(
                history.dtype
            )

        # If we are making any predictions, do them here
        output: Optional[Tensor] = None
        attention_scores: Optional[Tensor] = None

        history_seq_len: Optional[Tensor] = None

        if predict_mask.any():
            # history_seq, history_seq_len = get_history_seq(
            #     history, predict_mask, dialogue_timestep
            # )

            history_seq = history[predict_mask]
            history_seq_len = dialogue_timestep[predict_mask]

            outputs: List[Tensor] = []
            attention_scores: List[Tensor] = []

            # Loop over all the decoders and their related hidden states.
            # In practice, there will either be `n` decoders for each of the `n`
            # output features, or there will be 1 decoder for all `n` output features.
            decoder_num = 0
            for h, decoder in zip(decoder_hidden, self.decoders):
                decoder_num += 1
                # At minimum, the attention context vector contains the current
                # hidden state of the decoder
                attention_context: List[Tensor] = [get_hidden_vector(h)[predict_mask]]

                additional_decoder_input: List[Tensor] = []

                # We can optionally add more information to the attention context vector
                # and to the decoder input, under the assumption that including more
                # information will lead to better, more targeted attention scores or better
                # output predictions.
                # TODO: Make this controllable via a constructor flag
                if embeddings_encoded is not None:
                    embeddings_predict = embeddings_encoded[predict_mask[batch_mask]]

                    # Include the transcript of the upcoming turn in the context vector
                    attention_context.append(embeddings_predict)

                    # Include the transcript of the upcoming turn in the decoder input
                    additional_decoder_input.append(embeddings_predict)

                attention_context: Tensor = torch.cat(attention_context, dim=-1)

                decoded, scores = decoder(
                    encoded_seq=history_seq,
                    encoded_seq_len=history_seq_len,
                    attention_context=attention_context,
                    additional_decoder_input=additional_decoder_input,
                    hidden=h,
                    batch_idx=predict_mask,
                )

                outputs.append(decoded)
                attention_scores.append(scores.squeeze(-1))

            output = torch.cat(outputs, dim=-1)
            attention_scores: Tensor = nn.utils.rnn.pad_sequence(
                attention_scores, batch_first=True
            )
            attention_scores = torch.permute(attention_scores, (1, 2, 0))

        return output, attention_scores, history_seq_len
