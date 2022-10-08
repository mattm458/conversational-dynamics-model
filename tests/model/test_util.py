import unittest
from model import util
from torch import Tensor
import torch


class TestGetEmbeddingsSubsequence(unittest.TestCase):
    def test_get_embeddings_subsequence(self):
        # Tests the ability to retrieve an embedding subsequence from a larger
        # embedding tensor.

        # Create fake embeddings and subsequence lengths
        embeddings_all = torch.arange(10).repeat(3, 1).unsqueeze(2).repeat(1, 1, 3)
        start = torch.LongTensor([2, 4, 5])
        end = torch.LongTensor([7, 5, 9])

        embeddings, embeddings_len = util.get_embeddings_subsequence(
            embeddings=embeddings_all, start=start, end=end
        )

        # Hardcode our expected output. Note the indices from the range tensor,
        # and how they align with our expectations from the start and end indices
        output = torch.tensor(
            [
                [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]],
                [[4, 4, 4], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [0, 0, 0]],
            ]
        )

        # Hardcode the lengths tensor. This shows how long each embedding
        # subsequence is, and should reflect the zero padding seen in the
        # output tensor above.
        output_len = torch.LongTensor([5, 1, 4])

        self.assertTrue(torch.equal(embeddings, output))
        self.assertTrue(torch.equal(embeddings_len, output_len))

    def test_get_embeddings_subsequence_zero(self):
        # Tests the ability to retrieve an embedding subsequence from a larger
        # embedding tensor, when some dialogues in the batch are over
        # and have no remaining embeddings.

        # Create fake embeddings and subsequence lengths
        embeddings_all = torch.arange(10).repeat(3, 1).unsqueeze(2).repeat(1, 1, 3)
        start = torch.LongTensor([2, 4, 0])
        end = torch.LongTensor([7, 5, 0])

        embeddings, embeddings_len = util.get_embeddings_subsequence(
            embeddings=embeddings_all, start=start, end=end
        )

        # Hardcode our expected output. Note the sequence of zeros at the end,
        # and how this reflects the 0 start/end indices above.
        output = torch.tensor(
            [
                [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]],
                [[4, 4, 4], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        )

        # Hardcode the lengths tensor. The last length shows that there is
        # no data in the last embedding subsequence.
        output_len = torch.LongTensor([5, 1, 0])

        self.assertTrue(torch.equal(embeddings, output))
        self.assertTrue(torch.equal(embeddings_len, output_len))

    def test_get_embeddings_subsequence_start(self):
        # Tests the ability to retrieve an embedding subsequence from a larger
        # embedding tensor at the beginning when the start indices are all 0

        # Create fake embeddings and subsequence lengths
        embeddings_all = torch.arange(10).repeat(3, 1).unsqueeze(2).repeat(1, 1, 3)
        start = torch.LongTensor([0, 0, 0])
        end = torch.LongTensor([5, 3, 2])

        embeddings, embeddings_len = util.get_embeddings_subsequence(
            embeddings=embeddings_all, start=start, end=end
        )

        # Hardcode our expected output. Note the sequence of zeros at the end,
        # and how this reflects the 0 start/end indices above.
        output = torch.tensor(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        )

        # Hardcode the lengths tensor. The last length shows that there is
        # no data in the last embedding subsequence.
        output_len = torch.LongTensor([5, 3, 2])

        self.assertTrue(torch.equal(embeddings, output))
        self.assertTrue(torch.equal(embeddings_len, output_len))


class TestGetHiddenVector(unittest.TestCase):
    def test_get_hidden_vector_1_layer(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=1,
            device="cpu",
        )

        # Change the contents for testing
        hidden[-1][0][0] = 100.0

        hidden_vector = util.get_hidden_vector(hidden=hidden, last=False)

        self.assertIs(type(hidden_vector), Tensor)

        # get_hidden_vector should return a 3x10 tensor
        self.assertEqual(hidden_vector.shape, (3, 10))
        self.assertTrue(torch.equal(hidden_vector, hidden[-1][0]))

    def test_get_hidden_vector_1_layer_last(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=1,
            device="cpu",
        )

        # Change the contents for testing
        hidden[-1][0][0] = 100.0

        hidden_vector = util.get_hidden_vector(hidden=hidden, last=True)

        self.assertIs(type(hidden_vector), Tensor)

        # get_hidden_vector should return a 3x10
        self.assertEqual(hidden_vector.shape, (3, 10))
        self.assertTrue(torch.equal(hidden_vector, hidden[-1][0]))

    def test_get_hidden_vector_multilayer_last(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=2,
            device="cpu",
        )

        # Mark the last layer so we can recognize it later
        hidden[-1][0][0] = 100.0

        hidden_vector = util.get_hidden_vector(hidden=hidden, last=True)

        self.assertIs(type(hidden_vector), Tensor)

        # get_hidden_vector should return a 3x10 tensor that is the
        # last tensor in hidden_vector
        self.assertEqual(hidden_vector.shape, (3, 10))
        self.assertTrue(torch.equal(hidden_vector, hidden[-1][0]))

    def test_get_hidden_vector_multilayer(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=2,
            device="cpu",
        )

        # Change the contents for testing
        hidden[-1][0][0] = 100.0

        hidden_vector = util.get_hidden_vector(hidden=hidden, last=False)

        # Replicate the expected behavior of get_hidden_vector
        expected = torch.cat([h for h, _ in hidden], dim=1)

        self.assertIs(type(hidden_vector), Tensor)

        # get_hidden_vector should return a 3x20 tensor
        self.assertEqual(hidden_vector.shape, (3, 20))
        self.assertTrue(torch.equal(expected, hidden_vector))


class TestInitHiddenLayer(unittest.TestCase):
    def test_init_hidden_1_layer(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=1,
            device="cpu",
        )

        self.assertIs(type(hidden), list)
        self.assertEqual(len(hidden), 1)

        h1 = hidden[0]

        self.assertIs(type(h1), tuple)
        self.assertEqual(len(h1), 2)

        h, c = h1

        self.assertIs(type(h), Tensor)
        self.assertIs(type(c), Tensor)

        self.assertEqual(h.shape, (3, 10))
        self.assertEqual(c.shape, (3, 10))

    def test_init_hidden_multilayer(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=2,
            device="cpu",
        )

        self.assertIs(type(hidden), list)
        self.assertEqual(len(hidden), 2)

        for hc in hidden:
            self.assertIs(type(hc), tuple)
            self.assertEqual(len(hc), 2)

            h, c = hc

            self.assertIs(type(h), Tensor)
            self.assertIs(type(c), Tensor)

            self.assertEqual(h.shape, (3, 10))
            self.assertEqual(c.shape, (3, 10))
