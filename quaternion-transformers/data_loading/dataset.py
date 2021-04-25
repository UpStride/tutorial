import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

tf_preprocessing = tf.keras.layers.experimental.preprocessing


class PtEnDataset:
    """TED Talks Tranlsation Portuguese-to-English dataset class.

    Each sample is a pair of strings where the same sentence is expressed
    in both Portuguese and English.
    Example sample:
    data = "este é um problema que temos que resolver."
    label = "this is a problem we have to solve ."

    """

    def __init__(self):
        super(PtEnDataset, self).__init__()

        data, _ = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                            as_supervised=True)

        self.train_set, self.val_set = data['train'], data['validation']
        self.__load_tokenizer()

    def __load_tokenizer(self):
        """Load the tokenizers for both the Portuguese and the English language

        Private method to load the tokenizers for both languages and create an
        attribute for it.

        """

        model_name = "ted_hrlr_translate_pt_en_converter"
        file_name = model_name + ".zip"
        url = "https://storage.googleapis.com/download.tensorflow.org/models/"
        url += file_name

        tf.keras.utils.get_file(
                file_name,
                url,
                cache_dir='.', cache_subdir='', extract=True
        )

        self.tokenizers = tf.saved_model.load(model_name)

    def __tokenize_pairs(self, pt, en):
        """Return tokenized pairs of sentences.

        Private method that takes a sample composed of a sentente in Portuguese
        and its translation in English and tokenize + zero pad them.

        Args:
            pt (ragged tensor): Portuguese sentence
            en (ragged tensor): English sentence

        Return:
            pt (tensor): tokenized + zero padded Portuguese sentence
            en (tensor): tokenized + zero padded English sentence

        """

        pt = self.tokenizers.pt.tokenize(pt)
        # Convert from ragged to dense, padding with zeros.
        pt = pt.to_tensor()

        en = self.tokenizers.en.tokenize(en)
        # Convert from ragged to dense, padding with zeros.
        en = en.to_tensor()
        return pt, en

    def get_vocab_size(self):
        """
        Return the sizes of the Portuguese and the English vocabulary

        Return:
            (int): Portuguese vocabulary size
            (int): English vocabulary size

        """
        return self.tokenizers.pt.get_vocab_size(), \
            self.tokenizers.en.get_vocab_size()

    def get_tokenizers(self):
        """
        Return the Portuguese and the English tokenizers

        Return:
            (obj): Portuguese tokenizer
            (obj): English tokenizer

        """
        return self.tokenizers.pt, self.tokenizers.en

    def get_dataset_name(self):
        """
        Return the reference name of the dataset.

        Return:
            (str): name of the dataset.

        """
        return 'ted-talks_pt-en'

    def get_batched_data(self, batch_size, buffer_size=20000):
        """Return the train and test batch iterators

        Args:
            batch_size (int): train/test batch size
            buffer_size (int): buffer size for shuffling the train/test set

        Returns:
            train_batches (obj): train batch iterators
            val_batches (obj): test batch iterators

        """

        def make_batches(ds):
            return (
                    ds
                    .cache()
                    .shuffle(buffer_size)
                    .batch(batch_size)
                    .map(self.__tokenize_pairs,
                         num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE))

        train_batches = make_batches(self.train_set)
        val_batches = make_batches(self.val_set)

        return train_batches, val_batches


class ImdbDataset:
    """IMDb Reviews dataset class.

    Each sample is composed of an English text representing a movie review from
    the IMDb website and a binary label that indicates if the review represents
    a positive (1) or a negative (0) sentiment.
    Example sample:
    data = "This is the kind of film for a snowy Sunday afternoon when the rest
            of the world can go ahead with its own business as you descend into
            a big arm-chair and mellow for a couple of hours. Wonderful
            performances from Cher and Nicolas Cage (as always) gently row the
            plot along. There are no rapids to cross, no dangerous waters, just
            a warm and witty paddle through New York life at its best. A family
            film in every sense and one that deserves the praise it received."
    label = 1

    """

    def __init__(self):
        super(ImdbDataset, self).__init__()

        data, _ = tfds.load('imdb_reviews', with_info=True,
                            as_supervised=True)

        self.train_set, self.val_set = data['train'], data['test']
        self.__load_tokenizer()

    def __load_tokenizer(self):
        """Create the tokenizers for the movie review."""

        max_vocab_size = 8000
        max_seq_len = 500
        tokenizers = tf_preprocessing.TextVectorization(
                                      max_tokens=max_vocab_size,
                                      output_sequence_length=max_seq_len)
        tokenizers.adapt(self.train_set.map(lambda text, label: text))
        self.tokenizers = tokenizers

    def __tokenize(self, text, label):
        """Return tokenized text and unchanged label.

        Args:
            text (ragged tensor): review text
            label (ragged tensor): binary sentiment label

        Return:
            (tensor): tokenized + zero padded Portuguese sentence
            (tensor): tokenized + zero padded English sentence

        """

        return self.tokenizers(text), label

    def get_vocab_size(self):
        """
        Return the sizes of the vocabulary

        Return:
            (int): vocabulary size

        """

        return len(self.tokenizers.get_vocabulary())

    def get_dataset_name(self):
        """
        Return the reference name of the dataset.

        Return:
            (str): name of the dataset.

        """

        return 'imdb_reviews'

    def get_batched_data(self, batch_size, buffer_size=10000):
        """Return the train and test batch iterators

        Args:
            batch_size (int): train/test batch size
            buffer_size (int): buffer size for shuffling the train/test set

        Returns:
            train_batches (obj): train batch iterators
            val_batches (obj): test batch iterators

        """

        def make_batches(ds):
            return (
                    ds
                    .shuffle(buffer_size)
                    .batch(batch_size)
                    .map(self.__tokenize,
                         num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE))

        train_batches = make_batches(self.train_set)
        val_batches = make_batches(self.val_set)

        return train_batches, val_batches


dataset_factory = {'translation': PtEnDataset,
                   'sentiment': ImdbDataset}
