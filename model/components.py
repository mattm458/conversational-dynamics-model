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


# class MelEncoder(nn.Module):
#     def __init__(self, out_dim):
#         super().__init__()

#         self.delta = torchaudio.transforms.ComputeDeltas()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 128, (5, 3), padding=(2, 1)),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
#             nn.Conv2d(128, 256, (5, 3), (1, 1), padding=(2, 1)),
#             nn.LeakyReLU(),
#             # nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
#             # nn.LeakyReLU(),
#             # nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
#             # nn.LeakyReLU(),
#             # nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
#             # nn.LeakyReLU(),
#             # nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
#             # nn.LeakyReLU(),
#         )

#         self.pre_rnn = nn.Sequential(nn.Linear(10 * 256, 768), nn.LeakyReLU())
#         self.rnn = nn.GRU(768, 128, batch_first=True, bidirectional=True)

#         self.frame_weight = nn.Linear(256, 256)
#         self.context_weight = nn.Linear(256, 1)

#         self.linear = nn.Sequential(
#             nn.Linear(256, 64), nn.LeakyReLU(), nn.Linear(64, out_dim)
#         )

#     def forward(self, mel_spectrogram, mel_spectrogram_len):
#         mel_spectrogram = mel_spectrogram.swapaxes(1, 2)

#         if mel_spectrogram.shape[2] % 2 == 1:
#             mel_spectrogram = torch.cat(
#                 [
#                     mel_spectrogram,
#                     torch.zeros(
#                         (mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1),
#                         device=mel_spectrogram.device,
#                     ),
#                 ],
#                 2,
#             )

#         d1 = self.delta(mel_spectrogram)
#         d2 = self.delta(d1)

#         x = torch.cat(
#             [mel_spectrogram.unsqueeze(1), d1.unsqueeze(1), d2.unsqueeze(1)], dim=1
#         ).swapaxes(2, 3)

#         output = self.conv(x)
#         output = output.permute(0, 2, 3, 1).reshape(
#             mel_spectrogram.shape[0], mel_spectrogram.shape[2], 256 * 10
#         )

#         output = self.pre_rnn(output)

#         output, _ = self.rnn(output)

#         att_output = self.frame_weight(output)
#         att_output = self.context_weight(att_output)

#         mask = torch.arange(att_output.shape[1], device=att_output.device)
#         mask = mask.unsqueeze(0).repeat(len(mel_spectrogram_len), 1)
#         mask = mask >= mel_spectrogram_len.unsqueeze(1)
#         mask = mask.unsqueeze(2)

#         att_output = att_output.masked_fill(mask, 1e-4)
#         att_output = torch.softmax(att_output, 1)
#         output = (output * att_output).sum(1)

#         return self.linear(output)
