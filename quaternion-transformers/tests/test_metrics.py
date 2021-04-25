import unittest
import tensorflow as tf
import numpy as np

from networks.optimization import accuracy_function, loss_function, loss_object


class TestAccuracy(unittest.TestCase):
    def test_input_shape(self):
        with self.assertRaises(ValueError):

            pred = tf.constant(np.ones([1, 1, 1, 1]))
            real = tf.constant(np.ones([1, 1, 1]))
            accuracy_function(real, pred)

        with self.assertRaises(AssertionError):
            pred = tf.constant(np.ones([1, 1, 1, 1]))
            real = pred
            accuracy_function(real, pred)

            pred = tf.constant(np.ones([1, 1, 1]))
            real = pred
            accuracy_function(real, pred)

    def test_sequential_output(self):
        real = tf.constant([[1, 1, 2, 2, 0, 0]]*4, dtype=tf.int64)
        pred = tf.one_hot(real, 3)
        self.assertEqual(accuracy_function(real, pred), 1.0)

        pred = tf.one_hot([[2, 2, 1, 1, 0, 0]]*4, 3)
        self.assertEqual(accuracy_function(real, pred), 0.0)

        pred = tf.one_hot([[1, 1, 1, 1, 0, 0]]*4, 3)
        self.assertEqual(accuracy_function(real, pred), 0.5)

    def test_non_sequantial_output(self):
        real = tf.constant([0, 0, 1, 1], dtype=tf.int64)
        pred = tf.one_hot(real, 2)
        self.assertEqual(accuracy_function(real, pred), 1.0)

        pred = tf.one_hot([1, 1, 0, 0], 2)
        self.assertEqual(accuracy_function(real, pred), 0.0)

        pred = tf.one_hot([1, 1, 1, 1], 2)
        self.assertEqual(accuracy_function(real, pred), 0.5)


class TestLoss(unittest.TestCase):
    def expected_sequential_loss(self, real, pred, depth):
        ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        real = tf.one_hot(real, depth)
        pred = tf.nn.softmax(pred)
        return ce_loss(real, pred)

    def test_input_shape(self):
        with self.assertRaises(ValueError):

            pred = tf.constant(np.ones([1, 1, 1, 1]))
            real = tf.constant(np.ones([1, 1, 1]))
            loss_function(real, pred)

        with self.assertRaises(AssertionError):
            pred = tf.constant(np.ones([1, 1, 1, 1]))
            real = pred
            loss_function(real, pred)

            pred = tf.constant(np.ones([1, 1, 1]))
            real = pred
            loss_function(real, pred)

    def test_sequential_output(self):
        depth = 3

        real = tf.constant([[1, 1, 2, 2, 0, 0]]*4, dtype=tf.int64)
        real_no_padding = tf.constant([[1, 1, 2, 2]]*4, dtype=tf.int64)
        pred_no_padding = tf.one_hot([[1, 1, 2, 2]]*4, 3)
        pred = tf.one_hot([[1, 1, 2, 2, 0, 0]]*4, 3)
        self.assertAlmostEqual(loss_function(real, pred).numpy(),
                               self.expected_sequential_loss(real_no_padding, pred_no_padding, depth).numpy(),
                               places=6)

        pred = tf.one_hot([[2, 2, 1, 1, 0, 0]]*4, 3)
        pred_no_padding = tf.one_hot([[2, 2, 1, 1]]*4, 3)
        self.assertAlmostEqual(loss_function(real, pred).numpy(),
                               self.expected_sequential_loss(real_no_padding, pred_no_padding, depth).numpy(),
                               places=6)

        pred = tf.one_hot([[1, 1, 1, 1, 0, 0]]*4, 3)
        pred_no_padding = tf.one_hot([[1, 1, 1, 1]]*4, 3)
        self.assertAlmostEqual(loss_function(real, pred).numpy(),
                               self.expected_sequential_loss(real_no_padding, pred_no_padding, depth).numpy(),
                               places=6)

    def test_non_sequantial_output(self):
        depth = 2

        real = tf.constant([0, 0, 1, 1], dtype=tf.int64)
        pred = tf.one_hot(real, 2)
        self.assertAlmostEqual(loss_function(real, pred).numpy(),
                         self.expected_sequential_loss(real, pred, depth).numpy(),
                         places=6)

        pred = tf.one_hot([1, 1, 0, 0], 2)
        self.assertAlmostEqual(loss_function(real, pred).numpy(),
                         self.expected_sequential_loss(real, pred, depth).numpy(),
                         places=6)

        pred = tf.one_hot([1, 1, 1, 1], 2)
        self.assertAlmostEqual(loss_function(real, pred).numpy(),
                         self.expected_sequential_loss(real, pred, depth).numpy(),
                         places=6)


if __name__ == "__main__":
    unittest.main()
