from collections import defaultdict
from os import path
from typing import Dict, List, Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn
from torch.utils.data import Dataset


class ConversationDataset(Dataset):
    def __init__(
        self,
        ses_ids: List[int],
        features_dir: str,
        embeddings_dir: str,
        speech_feature_keys: List[str] = [
            "pitch_mean_norm",
            "pitch_range_norm",
            "intensity_mean_norm",
            "jitter_norm",
            "shimmer_norm",
            "nhr_norm",
            "rate_norm",
        ],
    ):
        super().__init__()

        self.ses_ids = ses_ids
        self.features_dir = features_dir
        self.embeddings_dir = embeddings_dir
        self.speech_feature_keys = speech_feature_keys

    def __len__(self) -> int:
        return len(self.ses_ids)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        # Get the session ID associated with this index and the session data
        ses_id = self.ses_ids[idx]
        ses_data = torch.load(path.join(self.features_dir, f"{ses_id}.pt"))

        # Find out which of the speakers in the dialogue (A or B) is considered the
        # "them" speaker - the human half of the conversation that the model is attempting
        # to respond to
        speaker_them: str = ses_data['speaker'][0]

        # Deal with embeddings - create a tensor of start/end indices for each
        # embeddings sequence in the session
        embeddings_len_X = torch.load(
            path.join(self.embeddings_dir, f"{ses_id}-embeddings-len.pt")
        )
        embeddings_idx = torch.cat(
            [torch.zeros(1, dtype=torch.long), torch.cumsum(embeddings_len_X, dim=0)]
        )
        embeddings_idx_start_X = embeddings_idx[:-1]
        embeddings_idx_end_X = embeddings_idx[1:]

        output_X = defaultdict(list)
        output_y = defaultdict(list)

        # Keep track of the number of "us" turns there are in total
        us_count = 0

        for i, speaker in enumerate(ses_data['speaker']):
            features_raw = [ses_data[f][i] for f in self.speech_feature_keys]

            if speaker == speaker_them:
                output_X["speech_features"].append(features_raw)
                output_X["dialogue_timestep"].append(i)
                output_X["speaker"].append([1.0, 0.0])
                output_X["embeddings_idx_start"].append(embeddings_idx_start_X[i])
                output_X["embeddings_idx_end"].append(embeddings_idx_end_X[i])
                output_X["embeddings_encode_mask"].append(True)
                output_X["feature_encode_mask"].append(True)
                output_X["predict_mask"].append(False)
                output_X["autoregress_mask"].append(False)
            else:
                us_count += 1

                output_X["speech_features"].append(
                    [0.0 for x in range(len(self.speech_feature_keys))]
                )
                output_X["speech_features"].append(features_raw)

                output_X["dialogue_timestep"].extend([i] * 2)
                output_X["speaker"].extend([[0.0, 1.0]] * 2)
                output_X["embeddings_idx_start"].extend([embeddings_idx_start_X[i]] * 2)
                output_X["embeddings_idx_end"].extend([embeddings_idx_end_X[i]] * 2)
                output_X["embeddings_encode_mask"].extend([True, True])
                output_X["feature_encode_mask"].extend([False, True])
                output_X["predict_mask"].extend([True, False])
                output_X["autoregress_mask"].extend([False, True])

                output_y["speech_features"].append(features_raw)

        output_X = dict(output_X)
        output_X["speech_features"] = FloatTensor(output_X["speech_features"])
        output_X["speech_features_len"] = LongTensor([len(output_X["speech_features"])])
        output_X["dialogue_timestep"] = LongTensor(output_X["dialogue_timestep"])
        output_X["speaker"] = FloatTensor(output_X["speaker"])
        output_X["feature_encode_mask"] = BoolTensor(output_X["feature_encode_mask"])
        output_X["embeddings_encode_mask"] = BoolTensor(
            output_X["embeddings_encode_mask"]
        )
        output_X["predict_mask"] = BoolTensor(output_X["predict_mask"])
        output_X["autoregress_mask"] = BoolTensor(output_X["autoregress_mask"])

        output_y = dict(output_y)
        output_y["speech_features"] = FloatTensor(output_y["speech_features"])
        output_y["speech_features_len"] = LongTensor([len(output_y["speech_features"])])
        output_y["us_count"] = LongTensor([us_count])

        # Embeddings -------------------------------------------------------------

        output_X["embeddings"] = torch.load(
            path.join(self.embeddings_dir, f"{ses_id}-embeddings-cat.pt")
        )
        output_X["embeddings_len"] = embeddings_len_X
        output_X["embeddings_idx_start"] = torch.LongTensor(
            output_X["embeddings_idx_start"]
        )
        output_X["embeddings_idx_end"] = torch.LongTensor(
            output_X["embeddings_idx_end"]
        )

        return (output_X, output_y)


def collate_fn(
    batch: List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

    speech_features_X = []
    speech_features_len_X = []
    dialogue_timestep_X = []
    speaker_X = []
    embeddings_X = []
    embeddings_len_X = []
    embeddings_idx_start_X = []
    embeddings_idx_end_X = []
    feature_encode_mask_X = []
    embeddings_encode_mask_X = []
    predict_mask_X = []
    autoregress_mask_X = []

    speech_features_y = []
    speech_features_len_y = []
    us_count_y = []

    for X, y in batch:
        speech_features_X.append(X["speech_features"])
        speech_features_len_X.append(X["speech_features_len"])
        dialogue_timestep_X.append(X["dialogue_timestep"])
        speaker_X.append(X["speaker"])
        embeddings_X.append(X["embeddings"])
        embeddings_len_X.append(X["embeddings_len"])
        embeddings_idx_start_X.append(X["embeddings_idx_start"])
        embeddings_idx_end_X.append(X["embeddings_idx_end"])
        feature_encode_mask_X.append(X["feature_encode_mask"])
        embeddings_encode_mask_X.append(X["embeddings_encode_mask"])
        predict_mask_X.append(X["predict_mask"])
        autoregress_mask_X.append(X["autoregress_mask"])

        speech_features_y.append(y["speech_features"])
        speech_features_len_y.append(y["speech_features_len"])
        us_count_y.append(y['us_count'])

    output_X = {
        "speech_features": nn.utils.rnn.pad_sequence(
            speech_features_X, batch_first=True
        ),
        "speech_features_len": torch.cat(speech_features_len_X),
        "dialogue_timestep": nn.utils.rnn.pad_sequence(
            dialogue_timestep_X, batch_first=True
        ),
        "speaker": nn.utils.rnn.pad_sequence(speaker_X, batch_first=True),
        "embeddings": nn.utils.rnn.pad_sequence(embeddings_X, batch_first=True),
        "embeddings_len": nn.utils.rnn.pad_sequence(embeddings_len_X, batch_first=True),
        "embeddings_idx_start": nn.utils.rnn.pad_sequence(
            embeddings_idx_start_X, batch_first=True
        ),
        "embeddings_idx_end": nn.utils.rnn.pad_sequence(
            embeddings_idx_end_X, batch_first=True
        ),
        "feature_encode_mask": nn.utils.rnn.pad_sequence(
            feature_encode_mask_X, batch_first=True
        ),
        "embeddings_encode_mask": nn.utils.rnn.pad_sequence(
            embeddings_encode_mask_X, batch_first=True
        ),
        "predict_mask": nn.utils.rnn.pad_sequence(predict_mask_X, batch_first=True),
        "autoregress_mask": nn.utils.rnn.pad_sequence(
            autoregress_mask_X, batch_first=True
        ),
    }
    output_y = {
        "speech_features": nn.utils.rnn.pad_sequence(
            speech_features_y, batch_first=True
        ),
        "speech_features_len": torch.cat(speech_features_len_y),
        "us_count": torch.cat(us_count_y)
    }

    return (output_X, output_y)