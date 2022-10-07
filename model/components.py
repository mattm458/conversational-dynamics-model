from typing import List, Tuple

import torch
from torch import Tensor, nn

from model.util import lengths_to_mask


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()

        self.rnn = nn.ModuleList(
            [nn.LSTMCell(in_dim, hidden_dim)]
            + [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_input: Tensor,
        hidden: List[Tuple[Tensor, Tensor]],
        mask: Tensor,
    ) -> Tensor:
        if len(hidden) != len(self.rnn):
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        x = encoder_input

        for i, rnn in enumerate(self.rnn):
            h, c = hidden[i]

            h_out, c_out = rnn(x, (h[mask], c[mask]))
            x = h_out

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

            h[mask] = h_out.type(h.dtype)
            c[mask] = c_out.type(h.dtype)

        return x


class Attention(nn.Module):
    def __init__(
        self, history_in_dim: int, context_dim: int, att_dim: int, dropout: float
    ):
        super().__init__()

        self.context_dim = context_dim

        self.history = nn.Linear(history_in_dim, att_dim, bias=False)
        self.context = nn.Linear(context_dim, att_dim, bias=False)
        self.v = nn.Linear(att_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, history: Tensor, context: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        history_att = self.history(history)
        context_att = self.context(context).unsqueeze(1)
        score = self.v(torch.tanh(history_att + context_att))

        score = score.masked_fill(mask, float("-inf"))
        score = torch.softmax(score, dim=1)
        score_out = score

        score = score.squeeze(-1).unsqueeze(1)

        self.dropout(score)
        att_applied = torch.bmm(score, history)
        att_applied = att_applied.squeeze(1)

        return att_applied, score_out.detach()


class Decoder(nn.Module):
    def __init__(
        self,
        attention_in_dim: int,
        attention_context_dim: int,
        attention_dim: int,
        attention_dropout: float,
        decoder_in_dim: int,
        decoder_dropout: float,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
    ):
        super().__init__()

        self.num_layers = num_layers

        self.attention: Attention = Attention(
            history_in_dim=attention_in_dim,
            context_dim=attention_context_dim,
            att_dim=attention_dim,
            dropout=attention_dropout,
        )

        self.rnn = nn.ModuleList(
            [nn.LSTMCell(decoder_in_dim, hidden_dim)]
            + [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout = nn.Dropout(decoder_dropout)

        linear_arr = [nn.Linear(hidden_dim, output_dim)]
        if activation == "tanh":
            print("Decoder: Tanh activation")
            linear_arr.append(nn.Tanh())

        self.linear = nn.Sequential(*linear_arr)

    def forward(
        self,
        encoded_seq: Tensor,
        encoded_seq_len: Tensor,
        attention_context: Tensor,
        additional_decoder_input: List[Tensor],
        hidden: List[Tuple[Tensor, Tensor]],
        batch_idx: Tensor,
    ):
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        encoded_att, scores = self.attention(
            encoded_seq,
            attention_context,
            lengths_to_mask(encoded_seq_len, encoded_seq.shape[1]),
        )

        x = torch.cat([encoded_att] + additional_decoder_input, dim=-1)

        for i, rnn in enumerate(self.rnn):
            h, c = hidden[i]

            h_out, c_out = rnn(x, (h[batch_idx], c[batch_idx]))
            x = h_out

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

            h[batch_idx] = h_out.type(h.dtype)
            c[batch_idx] = c_out.type(c.dtype)

        x = self.linear(x)

        return x, scores