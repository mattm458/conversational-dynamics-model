import tempfile
import unittest
from os import path

import dataset
import torch


class TestDataloader(unittest.TestCase):
    def test_dataloader(self):
        embeddings = torch.ones((10, 100))
        with tempfile.TemporaryDirectory() as dir:
            output = {"speaker": ["A", "B", "A"], "feature": [1.0, 0.0, 1.0]}
            torch.save(output, path.join(dir, "1.pt"))
            torch.save(embeddings, path.join(dir, "1-embeddings-cat.pt"))
            torch.save(torch.tensor([3, 2, 5]), path.join(dir, "1-embeddings-len.pt"))

            ds = dataset.ConversationDataset(
                ses_ids=[1],
                features_dir=dir,
                embeddings_dir=dir,
                speech_feature_keys=["feature"],
            )
            x, y = ds[0]

        self.assertTrue(
            torch.equal(
                x["speech_features"], torch.tensor([[1.0], [0.0], [0.0], [1.0]])
            )
        )
        self.assertTrue(
            torch.equal(x["dialogue_timestep"], torch.LongTensor([0, 1, 1, 2]))
        )
        self.assertTrue(
            torch.equal(
                x["speaker"],
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]),
            )
        )
        self.assertTrue(
            torch.equal(x["embeddings_idx_start"], torch.LongTensor([0, 3, 3, 5]))
        )
        self.assertTrue(
            torch.equal(x["embeddings_idx_end"], torch.LongTensor([3, 5, 5, 10]))
        )
        self.assertTrue(
            torch.equal(
                x["embeddings_encode_mask"], torch.BoolTensor([True, True, True, True])
            )
        )
        self.assertTrue(
            torch.equal(
                x["feature_encode_mask"], torch.BoolTensor([True, False, True, True])
            )
        )
        self.assertTrue(
            torch.equal(
                x["predict_mask"], torch.BoolTensor([False, True, False, False])
            )
        )
        self.assertTrue(
            torch.equal(
                x["autoregress_mask"], torch.BoolTensor([False, False, True, False])
            )
        )
        self.assertTrue(torch.equal(x["speech_features_len"], torch.LongTensor([4])))
        self.assertTrue(torch.equal(x["embeddings"], embeddings))
        self.assertTrue(torch.equal(x["embeddings_len"], torch.LongTensor([3, 2, 5])))
