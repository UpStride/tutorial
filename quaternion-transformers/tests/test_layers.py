import unittest
import tensorflow as tf
import numpy as np
import upstride.type2.tf.keras.layers as us_layers

from networks.layers import LayerNormalization, \
    EncoderLayer, DecoderLayer


class TestLayerNormalization(unittest.TestCase):
    """ Testing the LayerNormalization layer from layers.py"""

    def setUp(self):
        np.random.seed(3003)

    def test_output_shape(self):
        for _ in range(10):
            self.batch_size = 4 * np.random.randint(1, 30)
            self.feature_length = np.random.randint(1, 50)
            self.ln_real = LayerNormalization(type_layers=tf.keras.layers)
            self.ln_quat = LayerNormalization(type_layers=us_layers)

            self.x = tf.convert_to_tensor(np.random.rand(self.batch_size,
                                                         self.feature_length))
            self.y_real = self.ln_real(self.x)
            self.y_quat = self.ln_quat(self.x)
            self.assertTrue(tf.reduce_all(tf.shape(self.x) == tf.shape(self.y_real)))
            self.assertTrue(tf.reduce_all(tf.shape(self.x) == tf.shape(self.y_quat)))

    def test_idempotence(self):
        for _ in range(10):
            batch_size = 4 * np.random.randint(1, 30)
            feature_length = np.random.randint(1, 50)
            self.ln_real = LayerNormalization(type_layers=tf.keras.layers)
            self.ln_quat = LayerNormalization(type_layers=us_layers)

            x = tf.convert_to_tensor(np.random.rand(batch_size, feature_length))
            y_real = self.ln_real(x)
            y_quat = self.ln_quat(x)

            np.testing.assert_allclose(self.ln_real(y_real), y_real,
                                       rtol=1e-2, atol=1e-4)
            np.testing.assert_allclose(self.ln_quat(y_quat), y_quat,
                                       rtol=1e-2, atol=1e-4)


class TestEncoderDecoderLayer(unittest.TestCase):
    """ Testing EncoderLayer and DecoderLayer from layers.py"""

    def create_dicts(self, batch_size, d_model, num_heads, dff,
                     inp_seq_len, tar_seq_len):
        self.network_types = ["real", "mixed", "quaternion"]

        self.encoders = {net_type: EncoderLayer(net_type,
                                                d_model,
                                                num_heads,
                                                dff)
                         for net_type in self.network_types}

        self.decoders = {net_type: DecoderLayer(net_type,
                                                d_model,
                                                num_heads,
                                                dff)
                         for net_type in self.network_types}

        self.inp_shapes = {
            "real": (batch_size, inp_seq_len, d_model),
            "mixed": (batch_size*4, inp_seq_len, d_model//4),
            "quaternion": (batch_size*4, inp_seq_len, d_model//4)
        }

        self.tar_shapes = {
            "real": (batch_size, tar_seq_len, d_model),
            "mixed": (batch_size*4, tar_seq_len, d_model//4),
            "quaternion": (batch_size*4, tar_seq_len, d_model//4)
        }

    def setUp(self):
        np.random.seed(3003)

    def test_output_shape(self):
        for _ in range(10):
            num_heads = np.random.randint(1, 5)
            d_model = 4 * num_heads * np.random.randint(1, 10)
            dff = np.random.randint(1, 100)
            batch_size = np.random.randint(1, 50)

            inp_seq_len = np.random.randint(1, 50)
            tar_seq_len = np.random.randint(1, 50)

            self.create_dicts(batch_size, d_model,
                              num_heads, dff,
                              inp_seq_len, tar_seq_len)
            training = np.random.randint(0, 1)

            for net_type in self.network_types:
                inp = np.random.rand(*self.inp_shapes[net_type])
                tar = np.random.rand(*self.tar_shapes[net_type])
                enc = self.encoders[net_type]
                dec = self.decoders[net_type]

                enc_pad_mask = tf.convert_to_tensor(np.zeros(
                    (batch_size, 1, 1, inp_seq_len)),
                    dtype=tf.float32)
                combined_mask = tf.convert_to_tensor(np.zeros(
                    (batch_size, 1, tar_seq_len, tar_seq_len)),
                    dtype=tf.float32)

                enc_output = enc(inp, training, enc_pad_mask)
                dec_output, _, _ = dec(tar, enc_output, training,
                                       combined_mask, enc_pad_mask)
                self.assertTrue(tf.reduce_all(tf.shape(inp) == tf.shape(enc_output)))
                self.assertTrue(tf.reduce_all(tf.shape(tar) == tf.shape(dec_output)))


if __name__ == "__main__":
    unittest.main()
