import tensorflow as tf
import upstride.type2.tf.keras.layers as us_layers

from networks.layers import EncoderLayer, DecoderLayer, \
                            Encoder, Decoder


class Transformer(tf.keras.Model):
    """Transformer for sequence-to-sequence predictions"""

    def __init__(self, network_type, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size,
                 pe_input, pe_target, dropout_rate=0.1):
        """Initialize encoder and decoder block + final dense layer.

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
            target_vocab_size (int): vocabulary size of the target language
            pe_input (int): upper limit of the input sequence length
            pe_target (int): upper limit of the target sequence length
            dropout_rate (float): Rate of neurons of the 1st dense layer to
                drop during the training phase.

        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(network_type, num_layers, d_model, num_heads,
                               dff, input_vocab_size, pe_input, dropout_rate)

        self.decoder = Decoder(network_type, num_layers, d_model, num_heads,
                               dff, target_vocab_size, pe_target, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, padding_mask, look_ahead_mask):
        """Encoder-decoder network + final classification layers

        Args:
            x (tensor): encoder input tensor. x.shape=(batch_size, seq_len_x)
            tar (tensor): target tensor. tar.shape=(batch_size, seq_len_tar)
            training (bool): True = training phase, False = test/val phase
            look_ahead_mask (tensor): mask of the 1st MHA in the decoder layers
                with shape broadcastable to (..., seq_len_tar, seq_len_tar).
                Defaults to None.
            padding_mask (tensor): mask of the 2nd MHA with shape in the decoder
                layers and also the MHA in the encoder layers.
                broadcastable to (..., seq_len_x, seq_len_tar)
                Defaults to None.

        Returns:
            final_output (tensor): logits.
                shape == (batch_size, input_seq_len, target_vocab_size)
            attention_weights(dict[tensor]): dictionary containing the
                attention weights of the two MHA layers of all decoder layers.
        """

        # shape = (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
                tar, enc_output, training, look_ahead_mask, padding_mask)

        # shape = (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


class TransformerEncoder(tf.keras.Model):
    """Transformer for sequence-to-one predictions"""

    def __init__(self, network_type, num_layers, d_model, num_heads, dff,
                 input_vocab_size, n_classes, pe_input, dropout_rate):
        """Initialize encoder block, pooling layer and final dense layer.

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
            n_classes (int): number of classes.
            pe_input (int): upper limit of the input sequence length.
            dropout_rate (float): Rate of neurons of the 1st dense layer to
                drop during the training phase.

        """
        super(TransformerEncoder, self).__init__()

        self.encoder = Encoder(network_type, num_layers, d_model, num_heads,
                               dff, input_vocab_size, pe_input, dropout_rate)

        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.final_layer = tf.keras.layers.Dense(n_classes)

    def call(self, inp, tar, training, enc_padding_mask):
        """Encoder network + pooling + final classification layers

        Args:
            inp (tensor): encoder input tensor
                shape == (batch_size, seq_len_inp)
            tar (tensor): target tensor. shape == (batch_size, seq_len_tar)
            enc_padding_mask (tensor): mask of the MHA in the encoder layers
                broadcastable to (..., seq_len_x, seq_len_tar)
                Defaults to None

        Returns:
            final_output (tensor): logits
                shape == (batch_size, input_seq_len, n_classes)
        """

        # shape = (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # shape = (batch_size, d_model)
        pooled_enc = self.pooling(enc_output)

        # shape = (batch_size, tar_seq_len, n_classes)
        final_output = self.final_layer(pooled_enc)

        return final_output


def get_network(args, vocab_size):
    """Transformer network factory

    Return the right network base on the task and the other hyper-params.

    Args:
        args (obj): arguments containing all necessary information regarding
            the task and the network
        vocab_size (int/list[int]):
            - sequence-to-one task: input vocabulary size
            - sequence-to-sequence task: list of input/target vocabulary size

    Returns:
        (obj): keras model

    Raises:
        ValueError: If `args.task` neither 'sentiment' nor 'translation'
    """
    if args.task == 'translation':
        return Transformer(
                network_type=args.type,
                num_layers=args.num_layers,
                d_model=args.d_model,
                num_heads=args.num_heads,
                dff=args.dff,
                input_vocab_size=vocab_size[0],
                target_vocab_size=vocab_size[1],
                pe_input=1000,
                pe_target=1000,
                dropout_rate=args.dropout_rate)
    elif args.task == 'sentiment':
        return TransformerEncoder(
                network_type=args.type,
                num_layers=args.num_layers,
                d_model=args.d_model,
                num_heads=args.num_heads,
                dff=args.dff,
                input_vocab_size=vocab_size,
                n_classes=2,
                pe_input=5000,
                dropout_rate=args.dropout_rate)
    else:
        raise ValueError(f'Task {args.task} has no defined network.')
