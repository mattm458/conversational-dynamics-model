import unittest
from model import util
from torch import Tensor
import torch


class TestGetHiddenVector(unittest.TestCase):
    def test_get_hidden_vector_1_layer(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=1,
            device="cpu",
        )

        hidden_vector = util.get_hidden_vector(hidden=hidden, last=False)

        # get_hidden_vector should return a 3x10 tensor
        self.assertIs(type(hidden_vector), Tensor)
        self.assertEqual(hidden_vector.shape, (3, 10))

    def test_get_hidden_vector_1_layer_last(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=1,
            device="cpu",
        )

        hidden_vector = util.get_hidden_vector(hidden=hidden, last=True)

        # get_hidden_vector should return a 3x10
        self.assertIs(type(hidden_vector), Tensor)
        self.assertEqual(hidden_vector.shape, (3, 10))

    def test_get_hidden_vector_multilayer_last(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=2,
            device="cpu",
        )

        hidden[0][0][0] = 100.0

        hidden_vector = util.get_hidden_vector(hidden=hidden, last=True)

        # get_hidden_vector should return a 3x10 tensor that is the
        # last tensor in hidden_vector
        self.assertIs(type(hidden_vector), Tensor)
        self.assertEqual(hidden_vector.shape, (3, 10))
        self.assertTrue(torch.equal(hidden_vector, hidden[0][0]))

    def test_get_hidden_vector_multilayer(self):
        hidden = util.init_hidden(
            batch_size=3,
            hidden_dim=10,
            num_layers=2,
            device="cpu",
        )

        hidden_vector = util.get_hidden_vector(hidden=hidden, last=False)

        print(hidden_vector.shape)

        # get_hidden_vector should return a 3x20 tensor
        self.assertIs(type(hidden_vector), Tensor)
        self.assertEqual(hidden_vector.shape, (3, 20))


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
