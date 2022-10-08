from typing import List, Tuple

import torch
from torch import Tensor, device
from torch.nn import functional as F


def autoregress_feature(
    input_speech_features: Tensor,
    previous_output: Tensor,
    timestep_autoregress_mask: Tensor,
    teacher_forcing: float,
):
    batch_size, num_features = input_speech_features.shape
    device = input_speech_features.device

    teacher_forcing_mask = torch.rand((batch_size, num_features), device=device)
    teacher_forcing_mask = teacher_forcing_mask < teacher_forcing

    feature_autoregress_mask = (
        timestep_autoregress_mask.unsqueeze(1) * ~teacher_forcing_mask
    )

    feature_timestep_mask = feature_autoregress_mask[timestep_autoregress_mask]
    feature_autoregress_idx = torch.nonzero(feature_autoregress_mask, as_tuple=True)

    input_speech_features = input_speech_features.index_put(
        feature_autoregress_idx, previous_output[feature_timestep_mask]
    )

    return input_speech_features


def history_expand(
    history: Tensor, new_values: Tensor, batch_idx: Tensor, timestep_idx: Tensor
) -> Tensor:
    batch_size = history.shape[0]
    num_new_values = new_values.shape[0]

    if len(batch_idx) > batch_size:
        raise Exception(
            f"Number of new values ({num_new_values}) is greater than the history tensor batch size ({batch_size})!"
        )

    if batch_idx.max() > batch_size:
        raise Exception(
            f"Batch indices for history expansion contains a value larger than history batch size ({batch_idx.max()} > {batch_size}"
        )

    return history.index_put(indices=(batch_idx, timestep_idx), values=new_values)


def init_hidden(
    batch_size: int, hidden_dim: int, num_layers: int, device: device
) -> List[Tuple[Tensor, Tensor]]:
    # This function creates a list of hidden layers for a custom multilayer
    # LSTM. The output is a list of tuples (containing the hidden tensor and
    # cell state tensor), where each element in the list corresponds to a layer
    # in the LSTM. Element 0 is intended to be the lowest layer.
    return [
        (
            torch.zeros(batch_size, hidden_dim, device=device),
            torch.zeros(batch_size, hidden_dim, device=device),
        )
        for _ in range(num_layers)
    ]


def lengths_to_mask(lengths: Tensor, max_size: int) -> Tensor:
    mask = torch.arange(max_size, device=lengths.device)
    mask = mask.unsqueeze(0).repeat(len(lengths), 1)
    mask = mask >= lengths.unsqueeze(1)
    mask = mask.unsqueeze(2)

    return mask


def get_embeddings_subsequence(
    embeddings: Tensor,
    start: Tensor,
    end: Tensor,
) -> Tuple[Tensor, Tensor]:
    # This function accepts three tensors:
    #
    # `embeddings` contains a sequence of embeddings of size
    # [batch, sequence_len, embeddings_dim]
    #
    # `subsequence_start` contains a starting index of size
    # [batch, sequence_len]
    #
    # `subsequence_end` contains an ending index of size
    # [batch, sequence_len]
    #
    # The function cuts out a subsequence of the `embeddings` tensor
    # according to the boundaries defined by `subsequence_start`
    # and `subsequence_end`, and returns two tensors:
    #
    #   1. The subset of `embeddings` defined by the start and end points
    #      of size [batch, subsequence_len, embeddings_dim]. The tensor
    #      is zero-padded to account for differences in subsequence
    #      lengths across the batch.
    #   2. The lengths of each subsequence

    subsequence_start = torch.min(start[start > 0]) if torch.any(start > 0) else 0
    subsequence_end = torch.max(end[end > 0])

    start = torch.clip(start - subsequence_start, min=0)
    end = torch.clip(end - subsequence_start, min=0)

    lengths = end - start
    longest = lengths.max()

    embeddings = torch.split(embeddings[:, subsequence_start:subsequence_end], 1, dim=0)
    start = torch.split(start, 1, dim=0)
    end = torch.split(end, 1, dim=0)

    embeddings_subsequence = []
    for e, start, end in zip(embeddings, start, end):
        e = e.squeeze(0)[start:end]
        embeddings_subsequence.append(F.pad(e, (0, 0, 0, longest - e.shape[0])))

    return torch.stack(embeddings_subsequence), lengths


def get_hidden_vector(hidden: List[Tuple[Tensor, Tensor]], last: bool = True) -> Tensor:
    # This function creates a tensor from the hidden state of a custom multilayer LSTM
    # hidden state list. It can be run in one of two modes:
    #
    #   last = True:  Only return the last hidden layer (i.e., the hidden layer
    #                 associated with the last layer).
    #   last = False: Return a tensor containing the concatenation of the hidden
    #                 layer from every layer of the LSTM.
    if last or len(hidden) == 1:
        return hidden[-1][0]

    hs: List[Tensor] = []
    for h, c in hidden:
        hs.append(h)
    return torch.cat(hs, dim=-1)
