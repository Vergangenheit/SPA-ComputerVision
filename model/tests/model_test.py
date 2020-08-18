import unittest
from unittest.mock import patch
import tensorflow as tf
from model.model import VGGSPA
from configs.config import CFG


class ModelTest(tf.test.TestCase):
    def setUp(self):
        """we define our inputs that can be accessed by all methods
        Args: self
        Return: None"""
        super(ModelTest, self).setUp()
        self.vggspa = VGGSPA(CFG)

    def tearDown(self):
        """Dissolves the inputs created in setUp method"""
        pass

    @patch('model.model.ImageDataGenerator')
    def test_train_generator(self, mock_image_data_generator):
        """? extract a sample preprocessed image and check if it's been rescaled, sheared, flipped and zoomed ?"""
        self.vggspa.train_generator()
        mock_image_data_generator.assert_called()


if __name__ == "__main__":
    tf.test.main()
