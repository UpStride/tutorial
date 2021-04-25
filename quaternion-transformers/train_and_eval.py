import logging
import sys
import os
import time
import datetime
import numpy as np
import tensorflow as tf

from tabulate import tabulate

from data_loading.dataset import dataset_factory
from networks.optimization import OptimizationManager, \
                                  loss_function, accuracy_function
from networks.network import get_network
from arguments.parser import PARSER

# suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

args = PARSER.parse_args()

# Dataset
dataset = dataset_factory[args.task]()
vocab_size = dataset.get_vocab_size()
train_batches, test_batches = dataset.get_batched_data(args.batch_size)

# Network
transformer = get_network(args, vocab_size)

# Checkpoint management
exp_name = args.exp_name
checkpoint_path = f"./checkpoints/{dataset.get_dataset_name()}/{exp_name}"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint and args.from_pretrained:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

# Tensorboard management
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + exp_name + '/' + current_time + '/train'
test_log_dir = 'logs/' + exp_name + '/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# Optimization management
opt_manager = OptimizationManager(args.task, transformer, args.d_model,
                                  train_loss, train_accuracy,
                                  test_loss, test_accuracy)
train_step = opt_manager.get_train_step()
test_step = opt_manager.get_test_step()

for epoch in range(args.epochs):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_batches):
        train_step(inp, tar)

        if epoch == 0 and batch == 0:
            weights = transformer.trainable_weights
            tab_list = [[v.name, v.get_shape(), np.prod(v.get_shape())] for v in weights]
            n_params = np.sum([tl[2] for tl in tab_list])
            n_params_no_embed = np.sum([tl[2]
                                        for tl in tab_list[:-2]
                                        if ('embedding' not in tl[0])])
            print(tabulate(tab_list, headers=['Name', 'Shape', 'Params']))
            print(f"Number of trainable parameters: {n_params}")
            print("Number of trainable parameters w/o the embedding layers" +
                  f" and w/o the final dense layer: {n_params_no_embed}")
            del weights

        if batch % 50 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} \
                Loss {train_loss.result():.4f} \
                Accuracy {train_accuracy.result():.4f}')

    # Log training metric on tensorboard
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    # Evaluation
    for inp, tar in test_batches:
        test_step(inp, tar)
    print(f'--- Test Metrics: Epoch {epoch + 1} \
    Loss {test_loss.result():.4f} \
    Accuracy {test_accuracy.result():.4f} ---')

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} \
          Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
