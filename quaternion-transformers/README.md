# Quaternion Transformers

Quaternion Transformers provides a simple codebase that showcases how to implement and use a quaternion-valued transformer networks for the popular tasks of sentiment analysis and neural machine translation.

## Table of Contents
- [Quaternion Transformers](#quaternion-transformers)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Supported tasks/datasets](#supported-tasksdatasets)
  - [Supported network architectures](#supported-network-architectures)
  - [Usage](#usage)
    - [Train and evaluate](#train-and-evaluate)
    - [Compute the BLEU score](#compute-the-bleu-score)
    - [Run unit tests](#run-unit-tests)
  - [References](#references)

## Description 

Our formulation of quaternion transformers is based on the work of Tay et al. [2], which we highly recommend reading before moving forward with this example. For the sake of convenience, here are the main point to keep in mind:
* the *full quaternion transformer* formulation consists in converting all layers of the encoder and the decoder of the transformer [1] from real-valued to quaternion-valued;
* the *partial quaternion transformer* formulation consists in converting all the multi-head attention layers of the encoder and the decoder of the transformer [1] from real-valued to quaternion-valued, while keeping the feed-forward layers real-valued;
* the conversion of a layer from real-valued to quaternion-valued is done by substituting scalar with quaternions and the dot product operation with the hamilton product operation.

<p style="text-align:center;"><img src="transformer.png" width="50%">

This code has been developped starting from the Tensorflow tutorial provided in [this](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb) colab notebook

## Supported tasks/datasets

* Portuguese to English translation on the [TED Talks datasets](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en)
* Sentiment Analysis on the [IMDb reviews dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews)

## Supported network architectures

* Vanilla transformer with encoder-decoder architecture for many-to-many predictions, as described in the [1]
* Vanilla transformer encoder with global pooling for many-to-one predictions, as described in the [1]
* Full quaternion tranformer with encoder-decoder architecture for many-to-many predictions, as described in the [2]
* Full quaternion tranformer encoder architecture for many-to-one predictions, as described in the [2]
* Partial quaternion tranformer with encoder-decoder architecture for many-to-many predictions, as described in the [2]
* Partial quaternion tranformer encoder architecture for many-to-one predictions, as described in the [2]

## Usage

Before starting, please make sure to follow the instructions at the root of this repository to set up everything you need to run this example.

### Train and evaluate

To train and evaluate your transformer, run: 

`python train_and_eval.py @arguments/args_{translation/sentiment}.txt --type {real/quaternion/mixed}`

The option `@` is used to indicate the file containing the default hyper-parameters to run the translation on the TED Talks datasets (`translation`) or the sentiment analysis on the IMDb reviews dataset (`sentiment`); the option `--type` indicates whether to run the vanilla (`real`), the full quaternion (`quaternion`), or the partial quaternion (`mixed`) version of the transformer. 

Example: `python train_and_eval.py @arguments/args_translation.txt --type real`

As an alternative to directly modifying the argument files, it's possible to overwrite the default arguments in the files by adding them to the command line. 

Example: `python train_and_eval.py @arguments/args_translation.txt --type real --num_heads 1`

For a description of all the available arguments, run `python train_and_eval.py -h` .

### Compute the BLEU score

The most common metric to evaluate the performance of neural machine translation models is to use the [BLEU](https://en.wikipedia.org/wiki/BLEU) score. After completing the training of your transformer for machine translation, run:

`python compute_bleu.py @arguments/args_translation.txt --type {real/quaternion/mixed}`

This command will load the latest model checkpoint stored in the `checkpoints/` folder during the training process and use the trained network to compute the BLEU score on the test set of the TED Talks dataset.

Note1: make sure that the arguments related to the network architecture, including the `--type`, are the same between the training command and the BLEU computation command. So, if you have trained the network by running `python train_and_eval.py @arguments/args_translation.txt --type real`, you can compute the BLEU score by running `python compute_bleu.py @arguments/args_translation.txt --type real`.

Note2: the BLEU score only makes sense in the context of sequence-to-sequence prediction. If you try to run the `compute_bleu.py` script on the sentiment analysis task, the code will throw an error.

### Run unit tests

If you modify the code, it might be useful to run the existing unit tests to check the new implementation. Run:

`python -m unittest tests/run_tests.py`

In the `tests/` folder, you will also find a convenient bash script to check if the training script can still run without errors on all tasks/network types combinations. Run:

`source tests/roundcheck_script.sh`

## References
* [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 2017-Decem(Nips), 5999–6009.
* [2] Tay, Y., Zhang, A., Tuan, L. A., Rao, J., Zhang, S., Wang, S., Fu, J., & Hui, S. C. (2019). Lightweight and efficient neural natural language processing with quaternion networks. ArXiv.