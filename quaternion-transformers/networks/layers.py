import tensorflow as tf
import numpy as np
import upstride.type2.tf.keras.layers as us_layers

from networks.utils import positional_encoding, hamilton_product, \
                           real_to_quaternion, quaternion_to_real

class MultiHeadAttention(tf.keras.layers.Layer):
    """Keras layer implementing the multi-head attention operation

    Implementation of the multi-head attention (MHA) module with dot product
    attention proposed in (Vaswani at al., "Attention is all you need", 2017)
    """

    def __init__(self, type_layers, d_model, num_heads):
        """Define key attributes and the linear layers

        Args:
            type_layers (obj): type or layer that will be used in the MHA
                Options: 'tf.keras.layers' or 'upstride.type2.tf.keras.layers'
            d_model (int): feature size x number of heads
            num_heads (int): number or head (a.k.a. linear layers) in the MHA
        """

        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.type_layers = type_layers

        assert d_model % self.num_heads == 0
        assert num_heads > 0
        assert d_model > 0
        assert type(self.num_heads) == int
        assert type(self.d_model) == int

        self.depth = d_model // self.num_heads

        self.wq = type_layers.Dense(d_model)
        self.wk = type_layers.Dense(d_model)
        self.wv = type_layers.Dense(d_model)

        self.dense = type_layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Separate (d_model) dimension into (num_heads, depth)

        Args:
            x (tensor): projected query/key/value to split
                shape == (batch_size, seq_len, d_model)
            batch_size (int): batch size

        Return:
            (tensor): shape == (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask, is_quaternion=False):
        """Calculate the attention weights

        q, k, v must have matching leading dimensions. k, v must have matching
        penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look
        ahead), but it must be broadcastable for addition.

        Args:
            q (tensor): query shape == (batch_size, num_heads, seq_len_q, depth)
            k (tensor): key shape == (batch_size, num_heads, seq_len_k, depth)
            v (tensor): value shape == (batch_size, num_heads, seq_len_k, depth)
            mask (tensor): Float tensor with shape broadcastable
                    to (..., seq_len_q, seq_len_k). Defaults to None
            is_quaternion (bool): True if the q, k, and v are quaternions;
                false otherwise. Defaults to False

        Returns:
            output (tensor): result of the dot product attention computation;
                shape == (batch_size, num_heads, seq_len_q, depth)
            attention_weights (tensor): attention weights;
                shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        """

        assert type(is_quaternion)== bool

        # shape = (..., seq_len_q, seq_len_k)
        if is_quaternion:
            matmul_qk = hamilton_product(q, k)
        else:
            matmul_qk = tf.matmul(q, k, transpose_b=True)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            if is_quaternion:
                mask = tf.concat([mask]*4, axis=0)
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        # shape = # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, q, k, v, mask):
        """Core computation of the MHA layer.

        Note:
            It is worth noting that k and v must have the same shape, while q
            and k (and v) can have a different shape on the second axis. In
            other words, seq_len_k == seq_len_v, but seq_len_q CAN be different
            from seq_len_k (seq_len_q != seq_len_k).

        Args:
            q (tensor): query shape == (batch_size, seq_len_q, d_model)
            k (tensor): key shape == (batch_size, seq_len_k, d_model)
            v (tensor): value shape == (batch_size, seq_len_v, d_model)
            mask (tensor): Float tensor with shape broadcastable
                    to (..., seq_len_q, seq_len_k). Defaults to None

        Returns:
            output (tensor): layer's output;
                shape == (batch_size, seq_len_q, d_model)
            attention_weights (tensor): attention weights;
                shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # shape = ...
        # (batch_size, num_heads, seq_len_q, depth)
        # (batch_size, num_heads, seq_len_k, depth)
        # (batch_size, num_heads, seq_len_v, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask, False if self.type_layers == tf.keras.layers else True)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class LayerNormalization(tf.keras.layers.Layer):
    """Wrapper around LayerNormalization to manage quaternionic inputs."""

    def __init__(self, type_layers):
        """Define the layer type and keras' layers normalization as attributes.

        Args:
            type_layers (obj): type or layer that will be used in the MHA.
                Options: 'tf.keras.layers' or 'upstride.type2.tf.keras.layers'

        """
        super(LayerNormalization, self).__init__()
        self.type_layers = type_layers
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        """Core computation of the layer.

        Args:
            x (tensor): real- or quaternion-valued tensor
                if x is 'real': x.shape=(batch_size, seq_len, d_model)
                if x is 'quaternion':
                x.shape=(batch_size*4, target_seq_len, d_model//4)

        Returns:
            (tensor): shape == x.shape

        """
        if self.type_layers == tf.keras.layers:
            return self.ln(x)
        else:
            x = quaternion_to_real(x)
            x = self.ln(x)
            x = real_to_quaternion(x)
            return x


class FeedForward(tf.keras.layers.Layer):
    """Small feed-forward network with two dense layers."""

    def __init__(self, type_layers, d_model, dff, dropout_rate=0.1):
        """Initialize dense layers and dropout layer.

        Args:
            type_layers (obj): type or layer that will be used in the dense layers.
                Options: 'tf.keras.layers' or 'upstride.type2.tf.keras.layers'
            d_model (int): number of neurons on the 2nd dense layer.
            dff (int): number of neurons on the 1st dense layer.
            dropout_rate (float): Rate of neurons of the 1st dense layer to
                drop during the training phase.

        """
        super(FeedForward, self).__init__()
        self.dense1 = type_layers.Dense(dff, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = type_layers.Dense(d_model)

    def call(self, x):
        """Core computation of the layer.

        Args:
            x (tensor): real- or quaternion-valued tensor
                if x is 'real': x.shape=(batch_size, seq_len, d_model)
                if x is 'quaternion':
                x.shape=(batch_size*4, target_seq_len, d_model//4)

        Returns:
            (tensor): shape == x.shape

        """
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    """Core encoder module -> MHA + FFN with skip connections"""
    def __init__(self, network_type, d_model, num_heads, dff, dropout_rate=0.1):
        """Initialize MHA, dense, normalization and dropout layers.

        Args:
            network_type (str): String indicating the type of encoder layer.
                Options:
                - 'real': standard real-values layer
                - 'quaternion': all feature and weight tensors are quaternions
                - 'mixed': MHA-related tensors are quaternions and FFN-related
                           tensors are reals.
            d_model (int): number of neurons on the 2nd dense layer.
            num_heads (int): number or head (a.k.a. linear layers) in the MHA.
            dff (int): number of neurons on the 1st dense layer.
            dropout_rate (float): Rate of neurons of the 1st dense layer to
                drop during the training phase.

        """
        super(EncoderLayer, self).__init__()

        self.network_type = network_type

        type_layers_mha = tf.keras.layers
        type_layers_ffn = tf.keras.layers
        d_model_mha = d_model
        d_model_ffn = d_model
        if network_type != 'real':
            assert d_model % 4 == 0
            type_layers_mha = us_layers
            d_model_mha = d_model//4
        if network_type == 'quaternion':
            type_layers_ffn = us_layers
            d_model_ffn = d_model//4
            dff = dff//4


        self.mha = MultiHeadAttention(type_layers_mha, d_model_mha, num_heads)
        self.ffn = FeedForward(type_layers_ffn, d_model_ffn, dff)

        self.layernorm_mha = LayerNormalization(type_layers_mha)
        self.layernorm_ffn = LayerNormalization(type_layers_ffn)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """Core layer computation.

        The EncoderLayer assumes that the inputs 'x' and is in
        the correct format to be processed by the multi-head attention layer
        and it maintains this format for the output 'out2'. More specifically:
        - 'type'=='real', I/O shape=(batch_size, target_seq_len, d_model)
        - 'type'!='real', I/O shape=(batch_size*4, target_seq_len, d_model//4)

        Args:
            x (tensor): real- or quaternion-valued tensor
            training (bool): True = training phase, False = test/val phase
            mask (tensor): Float tensor with shape broadcastable
                    to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
            out2 (tensor): layer's output
        """

        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm_mha(x + attn_output)

        # In the 'mixed' strategy, the FFN is real-valued
        if self.network_type == 'mixed':
            out1 = quaternion_to_real(out1)

        # if network_type == 'real' or 'mixed':
        #       ffn_output.shape=(batch_size, target_seq_len, d_model)
        # if network_type == 'quaternion':
        #       ffn_output.shape=(batch_size*4, target_seq_len, d_model//4)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm_ffn(out1 + ffn_output)

        if self.network_type == 'mixed':
            out2 = real_to_quaternion(out2)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """Core decoder module -> MHA (x2) + FFN with skip connections"""

    def __init__(self, network_type, d_model, num_heads, dff, dropout_rate=0.1):
        """Initialize MHA, dense, normalization and dropout layers.

        Args:
            network_type (str): String indicating the type of encoder layer.
                Options:
                - 'real': standard real-values layer
                - 'quaternion': all feature and weight tensors are quaternions
                - 'mixed': MHA-related tensors are quaternions and FFN-related
                           tensors are reals.
            d_model (int): number of neurons on the 2nd dense layer.
            num_heads (int): number or head (a.k.a. linear layers) in the MHA.
            dff (int): number of neurons on the 1st dense layer.
            dropout_rate (float): Rate of neurons of the 1st dense layer to
                drop during the training phase.

        """
        super(DecoderLayer, self).__init__()

        self.network_type = network_type

        type_layers_mha = tf.keras.layers
        type_layers_ffn = tf.keras.layers
        d_model_mha = d_model
        d_model_ffn = d_model
        if network_type != 'real':
            assert d_model % 4 == 0
            type_layers_mha = us_layers
            d_model_mha = d_model//4
        if network_type == 'quaternion':
            type_layers_ffn = us_layers
            d_model_ffn = d_model//4
            dff = dff//4

        self.mha1 = MultiHeadAttention(type_layers_mha, d_model_mha, num_heads)
        self.mha2 = MultiHeadAttention(type_layers_mha, d_model_mha, num_heads)
        self.ffn = FeedForward(type_layers_ffn, d_model_ffn, dff)

        self.layernorm_mha1 = LayerNormalization(type_layers_mha)
        self.layernorm_mha2 = LayerNormalization(type_layers_mha)
        self.layernorm_ffn = LayerNormalization(type_layers_ffn)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        """Core layer computation.

        The DecoderLayer assumes that the inputs 'x' and 'enc_output' are in
        the correct format to be processed by the multi-head attention layer
        and it maintains this format for the output 'out3'. More specifically:
        - 'type'=='real', I/O shape=(batch_size, target_seq_len, d_model)
        - 'type'!='real', I/O shape=(batch_size*4, target_seq_len, d_model//4)

        Note:
            It is worth noting that look_ahead_mask is actually a combinaiton
            of a lookahead and a padding mask. See create_masks() in
            networks/utils.py .

        Args:
            x (tensor): real- or quaternion-valued decoder input tensor
            enc_output (tensor): encoder output tensor; similar to 'x'.
                Note that the 'seq_len' of 'x' and 'enc_output' can be different.
            training (bool): True = training phase, False = test/val phase
            look_ahead_mask (tensor): mask of the 1st MHA with shape
                broadcastable to (..., seq_len_x, seq_len_x).
                Defaults to None.
            padding_mask (tensor): mask of the 2nd MHA with shape
                broadcastable to (..., seq_len_x, seq_len_enc)
                Defaults to None.

        Returns:
            out3 (tensor): layer's output
            attn_weights_block1 (tensor): attention weights of the 1st MHA layer;
                shape == (batch_size, num_heads, seq_len_x, seq_len_x)
            attn_weights_block2 (tensor): attention weights of the 2nd MHA layer;
                shape == (batch_size, num_heads, seq_len_x, seq_len_enc)
        """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm_mha1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm_mha2(attn2 + out1)

        # In the 'mixed' strategy, the FFN is real-valued
        if self.network_type == 'mixed':
            out2 = quaternion_to_real(out2)

        # if network_type == 'real' or 'mixed':
        #       ffn_output.shape=(batch_size, target_seq_len, d_model)
        # if network_type == 'quaternion':
        #       ffn_output.shape=(batch_size*4, target_seq_len, d_model//4)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm_ffn(ffn_output + out2)

        if self.network_type == 'mixed':
            out3 = real_to_quaternion(out3)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    """Full encoder network"""

    def __init__(self, network_type, num_layers, d_model, num_heads, dff,
                 input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        """Initialize embedding, pos. encoding, encoder and dropout layers.

        Args:
            network_type (str): String indicating the type of encoder layer.
                Options:
                - 'real': standard real-values layer
                - 'quaternion': all feature and weight tensors are quaternions
                - 'mixed': MHA-related tensors are quaternions and FFN-related
                           tensors are reals.
            num_layers (int): number of encoder layers
            d_model (int): number of neurons on the 2nd dense layer.
            num_heads (int): number or head (a.k.a. linear layers) in the MHA.
            dff (int): number of neurons on the 1st dense layer.
            input_vocab_size (int): vocabulary size of the input language
            maximum_position_encoding (int): upper limit of the sequence length
            dropout_rate (float): Rate of neurons of the 1st dense layer to
                drop during the training phase.

        """
        super(Encoder, self).__init__()

        self.network_type = network_type

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                d_model if network_type == 'real' else d_model//4)

        self.enc_layers = [EncoderLayer(network_type, d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """Embedding + positional encoding + stack of encoding layers

        Args:
            x (tensor): encoder input tensor. x.shape=(batch_size, seq_len)
            training (bool): True = training phase, False = test/val phase
            mask (tensor): Float tensor with shape broadcastable
                    to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
            x (tensor): layer's output.
                shape == (batch_size, input_seq_len, d_model)
        """

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        if self.network_type != 'real':
            x = real_to_quaternion(x)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        if self.network_type != 'real':
            x = quaternion_to_real(x)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """Full decoder network"""

    def __init__(self, network_type, num_layers, d_model, num_heads, dff,
                 target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        """Initialize embedding, pos. encoding, encoder and dropout layers.

        Args:
            network_type (str): String indicating the type of encoder layer.
                Options:
                - 'real': standard real-values layer
                - 'quaternion': all feature and weight tensors are quaternions
                - 'mixed': MHA-related tensors are quaternions and FFN-related
                           tensors are reals.
            num_layers (int): number of encoder layers
            d_model (int): number of neurons on the 2nd dense layer.
            num_heads (int): number or head (a.k.a. linear layers) in the MHA.
            dff (int): number of neurons on the 1st dense layer.
            target_vocab_size (int): vocabulary size of the target language
            maximum_position_encoding (int): upper limit of the sequence length
            dropout_rate (float): Rate of neurons of the 1st dense layer to
                drop during the training phase.

        """

        super(Decoder, self).__init__()

        self.network_type = network_type

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                d_model if network_type == 'real' else d_model//4)

        self.dec_layers = [DecoderLayer(network_type, d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        """Embedding + positional encoding + stack of decoding layers

        Args:
            x (tensor): decoder input tensor. x.shape=(batch_size, seq_len)
            enc_output (tensor): real- or quaternion-valued encoder output tensor
                if x is 'real': x.shape=(batch_size, seq_len, d_model)
                if x is 'quaternion':
                x.shape=(batch_size*4, seq_len, d_model//4)
                Note that the 'seq_len' of 'x' and 'enc_output' can be different.
            training (bool): True = training phase, False = test/val phase
            look_ahead_mask (tensor): mask of the 1st MHA with shape
                broadcastable to (..., seq_len_x, seq_len_x).
                Defaults to None.
            padding_mask (tensor): mask of the 2nd MHA with shape
                broadcastable to (..., seq_len_x, seq_len_enc)
                Defaults to None.

        Returns:
            x (tensor): layer's output.
                shape == (batch_size, input_seq_len, d_model)
            attention_weights(dict[tensor]): dictionary containing the
                attention weights of the two MHA layers of all decoder layers.

        """

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        if self.network_type != 'real':
            x = real_to_quaternion(x)
            enc_output = real_to_quaternion(enc_output)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask,
                                                   padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        if self.network_type != 'real':
            x = quaternion_to_real(x)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights