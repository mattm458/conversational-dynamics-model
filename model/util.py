from typing import List, Tuple

import torch
from torch import Tensor, device
from torch.nn import functional as F


def init_hidden(
    batch_size: int, hidden_dim: int, num_layers: int, device: device
) -> List[Tuple[Tensor, Tensor]]:
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
    subsequence_start: Tensor,
    subsequence_end: Tensor,
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
    lengths = subsequence_end - subsequence_start
    longest = lengths.max()

    embeddings = torch.split(embeddings, 1, dim=0)
    subsequence_start = torch.split(subsequence_start, 1, dim=0)
    subsequence_end = torch.split(subsequence_end, 1, dim=0)

    embeddings_subsequence = []
    for e, start, end in zip(embeddings, subsequence_start, subsequence_end):
        e = e.squeeze(0)[start:end]
        embeddings_subsequence.append(F.pad(e, (0, 0, 0, longest - e.shape[0])))

    return torch.stack(embeddings_subsequence), lengths


def get_hidden_vector(hidden: List[Tuple[Tensor, Tensor]], last: bool = True) -> Tensor:
    if last or len(hidden) == 1:
        return hidden[-1][0]

    hs: List[Tensor] = []
    for h, c in hidden:
        hs.append(h)
    return torch.cat(hs, dim=-1)
