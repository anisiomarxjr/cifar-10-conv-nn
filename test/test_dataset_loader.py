import unittest
from dataset_loader import load_dataset
import numpy as np


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        (X, Y) = load_dataset()
        self.X = X
        self.Y = Y

    def test_tensor_dims_match(self):
        self.assertTrue(self.X.shape == (50000, 3, 32, 32))
        self.assertTrue(self.Y.shape == (50000, 1))

    def test_random_image_dims_match(self):
        rand = np.random.randint(0, 49999)
        self.assertEqual(self.X[rand].shape, (3, 32, 32), "Image dimensions do not match")


if __name__ == '__main__':
    unittest.main()