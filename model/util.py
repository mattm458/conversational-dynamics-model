from typing import List, Tuple

import torch
from torch import Tensor, device, nn


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
    embeddings_subsequence_start: Tensor,
    embeddings_subsequence_end: Tensor,
) -> Tuple[Tensor, Tensor]:
    # This function accepts three tensors:
    #
    # `embeddings` contains a sequence of embeddings of size
    # [batch, sequence_len, embeddings_dim]
    #
    # `embeddings_subsequence_start` contains a starting index of size
    # [batch, sequence_len]
    #
    # `embeddings_subsequence_end` contains an ending index of size
    # [batch, sequence_len]
    #
    # The function cuts out a subsequence of the `embeddings` tensor
    # according to the boundaries defined by `embeddings_subsequence_start`
    # and `embeddings_subsequence_end`, and returns two tensors:
    #
    #   1. The subset of `embeddings` defined by the start and end points
    #      of size [batch, subsequence_len, embeddings_dim]. The tensor
    #      is zero-padded to account for differences in subsequence
    #      lengths across the batch.
    #   2. The lengths of each subsequence
    batch_size = embeddings.shape[0]
    device = embeddings.device

    start = embeddings_subsequence_start.min()
    end = embeddings_subsequence_end.max()

    mask = torch.arange(start, end, device=device).repeat(batch_size, 1)
    mask = (mask >= embeddings_subsequence_start.unsqueeze(1)) * (
        mask < embeddings_subsequence_end.unsqueeze(1)
    )
    embeddings_subsequence_tensor = embeddings[:, start:end].masked_fill(
        ~mask.unsqueeze(2), 0.0
    )

    return (
        embeddings_subsequence_tensor,
        embeddings_subsequence_end - embeddings_subsequence_start,
    )


def get_hidden_vector(hidden: List[Tuple[Tensor, Tensor]], last: bool = True) -> Tensor:
    if last or len(hidden) == 1:
        return hidden[-1][0]

    hs: List[Tensor] = []
    for h, c in hidden:
        hs.append(h)
    return torch.cat(hs, dim=-1)
