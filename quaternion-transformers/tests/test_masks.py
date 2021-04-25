import unittest
import tensorflow as tf
import numpy as np
import random

from networks.utils import create_padding_mask, create_look_ahead_mask, create_masks

random.seed()

class TestPadding(unittest.TestCase):
    """Testing the function that creates a padding mask such that elements corresponding to sequence values are zero,
    and pad elements are one.

    Example:
        $ python -m unittest tests.test_masks.TestPadding -v
    """

    def setUp(self):
        self.y = tf.constant([[[[0., 0., 1., 1., 0.]]], [[[0., 0., 0., 1., 1.]]], [[[1., 1., 1., 0., 0.]]]],  dtype=tf.float32)

    def test_input_value_err(self):
        with self.assertRaises(ValueError):
            x = tf.constant([[7, 6, 0, 0], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.float32)
            create_padding_mask(x)

        with self.assertRaises(ValueError):
            x = tf.constant([[7, 6, 0, 0], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.int32)
            create_padding_mask(x)

        with self.assertRaises(ValueError):
            x = tf.constant([[[[5.0], [1]], [0]]], dtype=tf.float32)
            create_padding_mask(x)

    def test_input_invalid_arg(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant(3, dtype=tf.float32)
            create_padding_mask(x)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            create_padding_mask(4.5)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            create_padding_mask([])

    def test_input_arg_type_error(self):
        with self.assertRaises(TypeError):
            create_padding_mask("str")

    def test_output_shape(self):
        for _ in range(10):
            rand1 = random.randrange(1, 100)
            rand2 = random.randrange(1, 100)
            x = tf.constant(np.ones((rand1, rand2)), dtype=tf.float32)
            shape_eq = create_padding_mask(x).shape == (rand1, 1, 1, rand2)
            self.assertTrue(shape_eq)

    def test_output_values(self):
        x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.float32)
        values_eq = (create_padding_mask(x).numpy() == self.y.numpy()).all()
        self.assertTrue(values_eq)


class TestLookahead(unittest.TestCase):
    """Testing the function that creates a lookahead mask such that the mask is a
    matrix of ones where the upper triangular part is set to zero.

    Example:
        $ python -m unittest tests.test_masks.TestLookahead -v
    """

    def setUp(self):
        self.x = tf.constant(3, dtype=tf.int32)
        self.y = tf.constant([[0., 1., 1.], [0., 0., 1.], [0., 0., 0.]], dtype=tf.float32)

    def test_input_value_err(self):
        with self.assertRaises(ValueError):
            x = tf.constant([[[[5.0], [1]], [0]]], dtype=tf.float32)
            create_look_ahead_mask(x)

    def test_input_invalid_arg(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant([[7, 6]], dtype=tf.float32)
            create_look_ahead_mask(x)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant([[]], dtype=tf.float32)
            create_look_ahead_mask(x)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant([[4]], dtype=tf.float32)
            create_look_ahead_mask(x)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant([4], dtype=tf.float32)
            create_look_ahead_mask(x)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            create_look_ahead_mask([])

    def test_input_arg_type_error(self):
        with self.assertRaises(TypeError):
            create_padding_mask("str")

    def test_output_shape(self):
        for _ in range(10):
            rand1 = random.randrange(1,100)
            shape_eq = create_look_ahead_mask(rand1).shape == (rand1, rand1)
            self.assertTrue(shape_eq)

    def test_output_values(self):
        values_eq = (create_look_ahead_mask(self.x).numpy() == self.y.numpy()).all()
        self.assertTrue(values_eq)


class TestCombinedMasks(unittest.TestCase):
    """Testing the function that creates two types of masks and has two outputs accordingly:
    - padding mask for encoder and decoder
    - combined mask for the decoder - this masks combines the lookahead and the padding mask
    Here we test specifically for the second mask/output, as the first is trivial.

    Example:
        $ python -m unittest tests.test_masks.TestCombinedMasks -v
    """

    def setUp(self):
        self.y = tf.constant([[[[0.,1.,1.,1.,1.], [0.,0.,1.,1.,1.], [0.,0.,1.,1.,1.], [0.,0.,1.,0.,1.], [0.,0.,1.,0.,1.]]],
                         [[[0., 1., 1., 1., 1.], [0., 0., 1., 1., 1.], [0., 0., 1., 1., 1.], [0., 0., 1., 1., 1.], [0., 0., 1., 1., 0.]]],
                         [[[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 0., 1.], [1., 1., 1., 0., 0.]]]],
                        dtype=tf.float32)

    def test_input_value_error(self):
        with self.assertRaises(ValueError):
            x1 = tf.constant([[7, 6, 0, 0], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.float32)
            x2 = tf.constant([[7, 6, 0, 0], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.float32)
            create_masks(x1, x2)

        with self.assertRaises(ValueError):
            x = tf.constant([[[[5.0], [1]], [0]]], dtype=tf.float32)
            create_look_ahead_mask(x)

    def test_input_invalid_arg(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant([[7, 6]], dtype=tf.float32)
            create_look_ahead_mask(x)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant([[]], dtype=tf.float32)
            create_look_ahead_mask(x)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant([[4]], dtype=tf.float32)
            create_look_ahead_mask(x)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant([4], dtype=tf.float32)
            create_look_ahead_mask(x)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            create_look_ahead_mask([])

    def test_input_arg_type_error(self):
        with self.assertRaises(TypeError):
            create_padding_mask("str")

    def test_2nd_output_shape(self):
        for _ in range(10):
            rand1 = random.randrange(1, 100)
            rand2 = random.randrange(1, 100)
            rand3 = random.randrange(1, 100)
            x1 = tf.constant(np.ones((rand1, rand3)), dtype=tf.float32)
            x2 = tf.constant(np.ones((rand2, rand3)), dtype=tf.float32)
            shape_eq = create_masks(x1, x2)[1].shape == (rand2, 1, rand3, rand3)
            self.assertTrue(shape_eq)

    def test_2nd_output_values(self):
        x1 = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.float32)
        x2 = tf.constant([[1, 5, 0, 1, 0], [2, 3, 0, 0, 1], [0, 0, 0, 4, 5]], dtype=tf.float32)
        values_eq = (create_masks(x1, x2)[1].numpy() == self.y.numpy()).all()
        self.assertTrue(values_eq)


if __name__ == "__main__":
    unittest.main()
