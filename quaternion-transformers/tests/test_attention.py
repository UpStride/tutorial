import unittest
import tensorflow as tf
import numpy as np
import upstride.type2.tf.keras.layers as us_layers
from networks.utils import positional_encoding, hamilton_product, real_to_quaternion, quaternion_to_real
from networks.layers import MultiHeadAttention
import random


class TestMultiHeadAttention(unittest.TestCase):
    """Test creation of MultiHeadAttention.

    Example:
              $ python -m unittest tests.test_attention.TestMultiHeadAttention -v
        """

    def test_input_assert_error(self):
        with self.assertRaises(AssertionError):
            MultiHeadAttention(tf.keras.layers, 512, 9)

        with self.assertRaises(AssertionError):
            MultiHeadAttention(tf.keras.layers, 0, 1)

        with self.assertRaises(AssertionError):
            MultiHeadAttention(tf.keras.layers, 88.8, 11.1)

    def test_input_type_error(self):
        with self.assertRaises(TypeError):
            MultiHeadAttention(tf.keras.layers, "str", 8)

        with self.assertRaises(TypeError):
            MultiHeadAttention(tf.keras.layers, 512, "str")

    def test_input_zero_error(self):
        with self.assertRaises(ZeroDivisionError):
            MultiHeadAttention(tf.keras.layers, 1, 0)


class TestSplitHeads(unittest.TestCase):
    """Test function that splits the last dimension into (num_heads, depth).
    Output shape should be (batch_size, num_heads, seq_len, depth).

    Example:
          $ python -m unittest tests.test_attention.TestSplitHeads -v
    """

    def setUp(self):
        self.num_heads = 4
        self.d_model = 128
        self.MHA = MultiHeadAttention(tf.keras.layers, self.d_model, self.num_heads)
        self.batch_size = random.randrange(1, 100)
        self.seq_len = random.randrange(1, 100)
        self.x = tf.constant(np.ones((self.batch_size, self.seq_len, self.d_model)), dtype=tf.float32)

    def test_input_invalid_argument(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            x = tf.constant(np.ones((self.batch_size, self.seq_len)), dtype=tf.float32)
            self.MHA.split_heads(x, self.batch_size )

        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.MHA.split_heads("str", self.batch_size )

        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.MHA.split_heads(self.x, 64.5)

    def test_output_shape(self):
        out = self.MHA.split_heads(self.x, self.batch_size)
        shape_eq = out.shape == (self.batch_size, self.num_heads, self.seq_len, self.d_model//self.num_heads)
        self.assertTrue(shape_eq)


class TestScaledDotProduct(unittest.TestCase):
    """Test scaled_dot_product_attention function that calculates and outputs:
    - attention = attention weights * v
    - attention weights

      Example:
          $ python -m unittest tests.test_attention.TestScaledDotProduct -v
      """

    def setUp(self):
        self.num_heads = 4
        self.d_model = 128
        self.MHA = MultiHeadAttention(tf.keras.layers, self.d_model, self.num_heads)
        self.batch_size = random.randrange(0, 100)
        self.seq_len_1 = random.randrange(0, 100)
        self.seq_len_2 = random.randrange(0, 100)
        self.f_dim = random.randrange(0, 100)
        self.q = tf.constant(np.ones((self.batch_size, self.seq_len_1, self.f_dim)), dtype=tf.float32)
        self.k = tf.constant(np.ones((self.batch_size, self.seq_len_2, self.f_dim)), dtype=tf.float32)
        self.v = tf.constant(np.ones((self.batch_size, self.seq_len_2, self.f_dim)), dtype=tf.float32)
        self.mask = None
        self.is_quaternion = False

    def expectedOutput(self, q, k, v, mask, is_quaternion=False):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def test_input_value_error(self):
        with self.assertRaises(TypeError):
            self.MHA.scaled_dot_product_attention(self.q, self.k, self.v, "str", False)

    def test_input_assert_error(self):
        with self.assertRaises(AssertionError):
            self.MHA.scaled_dot_product_attention(self.q, self.k, self.v, self.mask, "str")

    def test_input_invalid_argument(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.MHA.scaled_dot_product_attention("self.q", self.k, self.v, self.mask, False)

    def test_output1_shape(self):
        o1, o2 = self.MHA.scaled_dot_product_attention(self.q, self.k, self.v, self.mask, False)
        shape_eq = o1.shape == (self.batch_size, self.q.shape[1], self.v.shape[2])
        self.assertTrue(shape_eq)

    def test_output2_shape(self):
        o1, o2 = self.MHA.scaled_dot_product_attention(self.q, self.k, self.v, self.mask, False)
        shape_eq = o2.shape == (self.batch_size, self.q.shape[1], self.k.shape[1])
        self.assertTrue(shape_eq)

    def test_output1_value(self):
        o1, o2 = self.MHA.scaled_dot_product_attention(self.q, self.k, self.v, self.mask, False)
        value_eq = (o1.numpy() == self.expectedOutput(self.q, self.k, self.v, self.mask, is_quaternion=False)[0].numpy()).all()
        self.assertTrue(value_eq)

    def test_output2_value(self):
        o1, o2 = self.MHA.scaled_dot_product_attention(self.q, self.k, self.v, self.mask, False)
        value_eq = (o2.numpy() == self.expectedOutput(self.q, self.k, self.v, self.mask, is_quaternion=False)[1].numpy()).all()
        self.assertTrue(value_eq)


if __name__ == "__main__":
    unittest.main()
