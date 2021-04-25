import logging
import sys
import os
import warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu

from data_loading.dataset import dataset_factory
from networks.optimization import OptimizationManager, \
                                  loss_function, accuracy_function
from networks.network import get_network
from arguments.parser import PARSER

# suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

args = PARSER.parse_args()

if args.task != 'translation':
    raise ValueError(f'The BLEU score can only be computed for the',
                     ' translation task.')

# Dataset
dataset = dataset_factory[args.task]()
vocab_size = dataset.get_vocab_size()
batch_size = args.batch_size
_, test_batches = dataset.get_batched_data(batch_size)
inp_tokenizer, tar_tokenizer = dataset.get_tokenizers()

# Network
transformer = get_network(args, vocab_size)

# Checkpoint management
exp_name = args.exp_name
checkpoint_path = f"./checkpoints/{dataset.get_dataset_name()}/{exp_name}"
if not os.path.exists(checkpoint_path):
    raise ValueError(f'Folder {checkpoint_path} does not exists.',
                     ' Please make sure you have a pretrained model.')
ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

# Optimization management
opt_manager = OptimizationManager(args.task, transformer, args.d_model,
                                  None, None, None, None)
autoreg_step = opt_manager.get_autoregressive_eval_step()

# Computing of the cumulative BLEU-{1..4} score
print("Calculating BLEU score...")

bleu_score = []
pbar = tqdm(test_batches)
for inp, tar in pbar:

    pred = autoreg_step(inp, tar_tokenizer)

    _, end_token = tar_tokenizer.tokenize([''])[0]

    for i in range(len(inp)):
        idx_end_of_prediction = tf.argmax(pred[i] == end_token)
        if idx_end_of_prediction == 0:
            # no end token found, the whole sequence is used as prediction
            idx_end_of_prediction = len(pred[i])-1
        pred_i = [pred[i][:idx_end_of_prediction+1]]
        pred_text = list(tar_tokenizer.detokenize(pred_i).numpy())[0].decode("utf-8").split(' ')
        inp_text = list(inp_tokenizer.detokenize(inp).numpy())[i].decode("utf-8").split(' ')
        tar_text = list(tar_tokenizer.detokenize(tar).numpy())[i].decode("utf-8").split(' ')

        # Compute bleu score...
        # while ignoring unnecessary warnings raised by sentence_bleu()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bleu_score.append(sentence_bleu([tar_text], pred_text))

    pbar.set_description(f"--- mean BLEU: {np.mean(bleu_score)}")


print(f'--- BLEU score: {np.mean(bleu_score)}')
