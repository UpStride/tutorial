import argparse

PARSER = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                 description="Vanilla/Quaternion Transformers")


PARSER.add_argument('--exp_name',
                    type=str,
                    default='default_exp_name',
                    help="""Name of the experiment """
                    + """(to index logs and checkpoints)""")

PARSER.add_argument('--from_pretrained',
                    type=bool,
                    default=False,
                    help="""Start the training from pretrained weights""")

PARSER.add_argument('--type',
                    type=str,
                    default='real',
                    choices=['real', 'quaternion', 'mixed'],
                    help="""Type of tranformer implementation. """
                    + """real: standard; """
                    + """quaternion: using quaternions in all layers; """
                    + """mixed: using quaternions in the attention layers.""")

PARSER.add_argument('--task',
                    type=str,
                    default='translation',
                    choices=['translation', 'sentiment'],
                    help="""Name of the task to tackle. """
                    + """Supported: translation, 'sentiment""")

PARSER.add_argument('--num_layers',
                    type=int,
                    default=4,
                    help="""Number of encoder/decoder layers.""")

PARSER.add_argument('--d_model',
                    type=int,
                    default=128,
                    help="""Dimension of the feature vectors in the """
                    + """multi-head attention layer. This number must """
                    + """be divisible by the number of heads.""")

PARSER.add_argument('--num_heads',
                    type=int,
                    default=8,
                    help="""Number of heads (linear projections) in the """
                    + """multi-head attention layer.""")

PARSER.add_argument('--dff',
                    type=int,
                    default=512,
                    help="""Dimension of the hidden feature vector in the """
                    + """feed forward block.""")

PARSER.add_argument('--epochs',
                    type=int,
                    default=20,
                    help="""Number of training epochs.""")

PARSER.add_argument('--batch_size',
                    type=int,
                    default=64,
                    help="""Training batch.""")

PARSER.add_argument('--dropout_rate',
                    type=float,
                    default=0.1,
                    help="""Dropout rate.""")
