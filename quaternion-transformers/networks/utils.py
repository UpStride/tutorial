import tensorflow as tf
import numpy as np


def real_to_quaternion(x):
    """Return reshaped tensor to simulate a quaternion from a real tensor

    Note:
        Assume that the convention is to to stack the components of the
        quaternion in the batch axis.

    Args:
        x (tensor): shape=(batch_size, ..., d_model)

    Returns:
        (tensor): shape=(batch_size*4, ..., d_model//4)
    """
    return tf.concat(tf.split(x, 4, axis=-1), axis=0)


def quaternion_to_real(x):
    """Return reshaped tensor to get a real tensor from a simulated quaternion

    Note:
        Assume that the convention is to to stack the components of the
        quaternion in the batch axis.

    Args:
        x (tensor): shape=(batch_size*4, ..., d_model//4)

    Returns:
        (tensor): shape=(batch_size, ..., d_model)
    """
    return tf.concat(tf.split(x, 4, axis=0), axis=-1)


def positional_encoding(max_position, d_model):
    """Return the tensor encoding the position of elements in a sequence

    Compute the positional encoding with a combination of sinusoidal functions
    as suggested in (Vaswani at al., "Attention is all you need", 2017)

    Note:
        When we add the positional encoding produced by this function to the
        input embedding, we only use the first 'seq_len' elements of axis=1,
        not the full 'max_position'. As an example, check the Encoder class.

    Args:
        max_position (int): maximum possible length of a sequence
        d_model (int): dimension of the feature vector

    Returns:
        (tensor): shape=(1, max_position, d_model)
    """

    # compute argument of sinusoidal functions
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(max_position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # add batch axis to the array
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# Masking

def create_padding_mask(seq):
    """Return a binary padding mask

    Take as input a zero-padded sequence and create a tensor mask such that all
    elements corresponsing to actual sequence values are zero, while the pad
    elements are set to one. This mask is used to make sure the padding is not
    used when computing the attention weights.

    Args:
        seq (tensor): shape == (batch_size, seq_len)

    Returns:
        (tensor): shape == (batch_size, 1, 1, seq_len)
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq_len):
    """Return a binary lookahead mask

    Create a lookahead mask for the attention weights such that the mask is a
    (seq_len x seq_len) tensor of ones with the upper triangular part is set to
    zero. This mask is used to make sure the 'future' elements in the sequence
    are not used when computing the attention weights.

    Args:
        seq_len (int): sequence length

    Returns:
        (tensor): square tensor of shape=(seq_len, seq_len) w/ the diagonal and
        the lower triangular part to zero and the rest to one
    """

    # band_part(x, -1, 0) sets the lower triangular part of x to zero
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask


def create_masks(inp, tar):
    """Create padding masks for both the encoder and the decoder layers.

    Args:
        inp (tensor): shape == (batch_size, input_seq_len)
        tar (tensor): shape == (batch_size, target_seq_len)

    Returns:
        padding_mask (tensor): shape == (batch_size, input_seq_len)
        combined_mask (tensor): target padding mask + lookahead mask
            shape == (batch_size, 1, target_seq_len, target_seq_len)

    """

    # Encoder/decoder padding mask used in the encoder's self-attention layer
    # and in the 2nd attention block in the decoder.
    padding_mask = create_padding_mask(inp)

    # Mask for the 1st attention block in the decoder.
    # It combines the lookahead and the padding mask such that an element that
    # is either a padding element or a future element is ignored.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return padding_mask, combined_mask