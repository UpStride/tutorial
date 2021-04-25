import tensorflow as tf
import time
from networks.utils import create_padding_mask, create_masks


# Learning rate scheduler & optimizer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Custom learning rate schedule

    First warm up phase and reciprocal square root descent afterwards.
    Takes into account the dimension of the feature space d_model.
    After the warmup, we have:
    lr = (1/d_model)^0.5 * (1/step)^0.5
    """
    def __init__(self, d_model, warmup_steps=4000):
        """
        Args:
            d_model (int): dimension of the feature vector
            warmup_steps (int): number of warmup steps
        """
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# Loss & Accuracy

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')


def loss_function(real, pred):
    """ Computes the cross entropy loss.

    It can work for a single classification or for sequences.
    In the case of sequences a mask is applied to ignore the loss in padded areas.

    Args:
        real (tensor): ground truth tensor, NOT one-hot encoded.
            shape == (batch_size,) or (batch_size, seq_len)
        pred (tensor): network prediction, one-hot encoded.
            shape == (batch_size, num_classes) or (batch_size, seq_len, num_classes)

    Returns:
        loss (tensor): The cross entropy loss
            shape == ()

    Raises:
        ValueError if the rank of the pred tensor is neither 2 nor 3
    """
    assert len(real.shape) == len(pred.shape)-1

    rank_pred = len(pred.shape)
    if rank_pred == 3:
        is_sequence = True
    elif rank_pred == 2:
        is_sequence = False
    else:
        raise ValueError('Predictons should have rank 2 or 3,'
                         f' but have rank = {rank_pred}')

    loss_ = loss_object(real, pred)

    if is_sequence:
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss = tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    else:
        loss = tf.reduce_mean(loss_)

    return loss


def accuracy_function(real, pred):
    """ Computes the accuracy.

    It can work for a single classification or for sequences.
    In the case of sequences a mask is applied to ignore the loss in padded areas.

    Args:
        real (tensor): ground truth tensor, NOT one-hot encoded.
            shape == (batch_size,) or (batch_size, seq_len)
        pred (tensor): network prediction, one-hot encoded.
            shape == (batch_size, num_classes) or (batch_size, seq_len, num_classes)

    Returns:
        final_accuracy (tensor): The accuracy.
            shape == ()

    Raises:
        ValueError if the rank of the pred tensor is neither 2 nor 3
    """

    assert len(real.shape) == len(pred.shape)-1

    rank_pred = len(pred.shape)
    if rank_pred == 3:
        is_sequence = True
    elif rank_pred == 2:
        is_sequence = False
    else:
        raise ValueError('Predictons should have rank 2 or 3,'
                         f' but have rank = {rank_pred}')

    accuracies = tf.equal(real, tf.argmax(pred, axis=-1))

    if is_sequence:
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        mask = tf.cast(mask, dtype=tf.float32)
        final_accuracy = tf.reduce_sum(tf.cast(accuracies, dtype=tf.float32))/tf.reduce_sum(mask)
    else:
        final_accuracy = tf.reduce_mean(tf.cast(accuracies, dtype=tf.float32))

    return final_accuracy


# Training step functions

train_step_sequence_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, ), dtype=tf.int64),
]


class OptimizationManager:
    """used to centralize optimization functions.

    Creates the training step and test step according to the model and task
    """
    def __init__(self, task, model, d_model,
                 train_loss, train_accuracy, test_loss, test_accuracy):
        """

        Args:
            task (str): name of the task. can be "translation" or "sentiment"
            model (tf.keras.Model): keras model
            d_model (int): dimension of the feature vector
            train_loss (func): function that will be called during each
                training step on the loss (ie "train_loss(loss)")
                is used to record the loss during the training
            train_accuracy (func): similar to train_loss but for accuracy
            test_loss (func): similar to train_loss but during the test phase
            test_accuracy (func): similar to test_loss but for accuracy
        """
        super(OptimizationManager, self).__init__()
        self.task = task
        self.d_model = d_model
        self.model = model
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy

        self.train_step_sequence_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            ]
        self.train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, ), dtype=tf.int64),
            ]
        self.optimizer = self.__get_optimizer()

    def __get_optimizer(self):
        """Initializes and returns the optimizer with the custom lr schedule"""
        learning_rate = CustomSchedule(self.d_model)
        return tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=0.9,
                                        beta_2=0.98,
                                        epsilon=1e-9)

    @tf.function(input_signature=train_step_sequence_signature)
    def __train_step_sequence(self, inp, tar):
        """ Computes one training step

        - computes forward step
        - computes the loss
        - apply gradients with the optimizer
        - records loss and accuracy
        Works when the target is also a sequence, i.e. in a translation task

        Args:
            inp (tensor): input sequences
                shape == (batch_size, inp_seq_len)
            tar (tensor): target sequences
                shape == (batch_size, tar_seq_len)
        """

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        padding_mask, combined_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.model(inp, tar_inp,
                                        True,
                                        padding_mask,
                                        combined_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tar_real, predictions))

    @tf.function(input_signature=train_step_signature)
    def __train_step(self, inp, tar):
        """ Computes one training step

        - computes forward step
        - computes the loss
        - apply gradients with the optimizer
        - records loss and accuracy
        Works when the target is a single class, i.e. in a sentiment analysis task

        Args:
            inp (tensor): input sequences
                shape == (batch_size, inp_seq_len)
            tar (tensor): target classes
                shape == (batch_size,)
        """

        enc_padding_mask = create_padding_mask(inp)

        with tf.GradientTape() as tape:
            predictions = self.model(inp, tar, True, enc_padding_mask)
            loss = loss_function(tar, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tar, predictions))

    @tf.function(input_signature=train_step_sequence_signature)
    def __test_step_sequence(self, inp, tar):
        """ Computes one test step

        - computes forward step
        - computes loss and accuracy
        - records loss and accuracy
        Works when the target is also a sequence, i.e. in a translation task

        Args:
            inp (tensor): input sequences
                shape == (batch_size, inp_seq_len)
            tar (tensor): target sequences
                shape == (batch_size, tar_seq_len)
        """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        padding_mask, combined_mask = create_masks(inp, tar_inp)

        predictions, _ = self.model(inp, tar_inp,
                                    True,
                                    padding_mask,
                                    combined_mask)
        loss = loss_function(tar_real, predictions)

        self.test_loss(loss)
        self.test_accuracy(accuracy_function(tar_real, predictions))

    @tf.function(input_signature=train_step_signature)
    def __test_step(self, inp, tar):
        """ Computes one test step

        - computes forward step
        - computes loss and accuracy
        - records loss and accuracy
        Works when the target is a single class, i.e. in a sentiment analysis task

        Args:
            inp (tensor): input sequences
                shape == (batch_size, inp_seq_len)
            tar (tensor): target classes
                shape == (batch_size,)
        """

        enc_padding_mask = create_padding_mask(inp)

        predictions = self.model(inp, tar, True, enc_padding_mask)
        loss = loss_function(tar, predictions)

        self.test_loss(loss)
        self.test_accuracy(accuracy_function(tar, predictions))

    def __autoregressive_eval_step(self, inp, tokenizer, max_len=500):
        """ Runs inference for the translation task without any groundtruth.

        Used to properly evaluate a model.
        For each sequence, the output of the model is recursively used
        to generate its next word.

        Args:
            inp (tensor): input sequences
                shape == (batch_size, inp_seq_len)
            tokenizer (obj): target language tokenizer
        """

        # start the prediction with the START token of the target language
        start, end = tokenizer.tokenize([''])[0]

        batch_size = tf.shape(inp)[0]
        out = tf.fill((batch_size, 1), start)
        end_tensor = tf.fill((batch_size, 1), end)

        ended_sequences = tf.fill((batch_size,), False)

        for i in range(max_len):
            padding_mask, combined_mask = create_masks(
                inp, out)

            # predictions.shape == (batch_size, tar_seq_len, vocab_size)
            # Note that 'tar_seq_len' increases progressively for a given
            # sample as the model sequentially generates the output words.
            predictions, _ = self.model(inp,
                                        out,
                                        False,
                                        padding_mask,
                                        combined_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_ids = tf.argmax(predictions, axis=-1)

            # identify the sequences that have outputted the end token
            # and update ended_sequences accordingly
            end_tokens = tf.squeeze(predicted_ids == end_tensor)
            ended_sequences = tf.logical_or(ended_sequences, end_tokens)

            # concatenate the predicted_ids to the output which is given to the
            # decoder as its input.
            out = tf.concat([out, predicted_ids], axis=-1)

            # return the result if all sequences have ended
            if tf.reduce_all(ended_sequences):
                break
        return out

    def get_train_step(self):
        if self.task == 'translation':
            return self.__train_step_sequence
        elif self.task == 'sentiment':
            return self.__train_step
        else:
            raise ValueError(f'No train_step() defined for task {self.task}.')

    def get_test_step(self):
        if self.task == 'translation':
            return self.__test_step_sequence
        elif self.task == 'sentiment':
            return self.__test_step
        else:
            raise ValueError(f'No test_step() defined for task {self.task}.')

    def get_autoregressive_eval_step(self):
        if self.task == 'translation':
            return self.__autoregressive_eval_step
        elif self.task == 'sentiment':
            return None
        else:
            raise ValueError(f'No test_step() defined for task {self.task}.')
