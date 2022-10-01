from re import S
from typing import Tuple

import torch
from torch import Tensor, nn

from model.components import Attention
from model.util import lengths_to_mask


class EmbeddingEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        encoder_out_dim: int,
        encoder_num_layers: int,
        encoder_dropout: float,
        attention_dim: int,
        attention_dropout: float,
    ):
        super().__init__()

        lstm_out_dim = encoder_out_dim // 2

        self.encoder_out_dim = encoder_out_dim
        self.encoder_num_layers = encoder_num_layers

        self.encoder = nn.LSTM(
            embedding_dim,
            lstm_out_dim,
            bidirectional=True,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
            batch_first=True,
        )

        self.attention = Attention(
            history_in_dim=encoder_out_dim,
            context_dim=encoder_out_dim,
            att_dim=attention_dim,
            dropout=attention_dropout,
        )

    def forward(self, encoder_in: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = encoder_in.shape[0]

        encoder_in = nn.utils.rnn.pack_padded_sequence(
            encoder_in,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        encoder_out, (h, _) = self.encoder(encoder_in)
        # Get the last hidden layer
        h = h.swapaxes(0, 1).reshape(batch_size, -1, self.encoder_out_dim)[:, -1]

        encoder_out, _ = nn.utils.rnn.pad_packed_sequence(encoder_out, batch_first=True)

        return self.attention(
            history=encoder_out,
            context=h,
            mask=lengths_to_mask(lengths, encoder_out.shape[1]),
        )
