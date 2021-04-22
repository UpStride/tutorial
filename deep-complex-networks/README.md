# Deep Complex Networks

Deep Complex Networks (DCN) provides a simple codebase that showcases the implementation of Deep Complex Networks using Tensorflow and python engine and train and evaluate on CIFAR datasets.

## Table of Contents
- [Deep Complex Networks](#deep-complex-networks)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Pre-requisites](#pre-requisites)
  - [Supported tasks/datasets](#supported-tasksdatasets)
  - [Supported network architectures](#supported-network-architectures)
  - [DCN Report](#dcn-report)
  - [Usage](#usage)
    - [Train and evaluate](#train-and-evaluate)
    - [How to modify the code](#how-to-modify-the-code)
  - [References](#references)

## Description 

Our implementation is based on the work of Trabelsi et al., which we highly recommend reading before moving forward with this example. 

We have conducted experiments that validate our implementation with Trabelsi et al [github][3]. Our implementation includes Complex BatchNormalization, Complex and independent initialization, Complex Convolution and learning the imaginary parts using a small resnet block.

The example will focus only on image classification task on CIFAR10 and CIFAR100 using the  Wide-Shallow architecture. 

## Pre-requisites

Please follow the instructions from classification-api [github][1] repository and setup the codebase. 

## Supported tasks/datasets

* Image classification on [CIFAR10 or CIFAR100][2] datasets

## Supported network architectures

We support all 3 network architectures from Trabelsi et al. Each one has trade-off between model width and depth while fixing the overall parameter (~ 1.7 M parameters). 

__Wide-Shallow__: Models use __less__ residual blocks and __more__ convolutional filters per layer.

* WSComplexNetTF
* WSComplexNetUpStride

__Deep-Narrow__: Models use __more__ residual blocks and __less__ convolutional filters per layer.

* DNComplexNetTF
* DNComplexNetUpStride

__In-Between__: Models use a good compromise between Wide-Shallow and Deep-Narrow.

* IBComplexNetTF
* IBComplexNetUpStride

## DCN Report

Our experiments to validate DCN can be found [here](.report-dcn.pdf). It's recommended to read the report on the experimental setup and processes used in our validation.

The DCN paper source code is available in this [github][3] repository.

__Caveats__:

Our python engine and classification-api has evolved at lot since the creation of the report, hence there are couple of differences between the experimental setup from the report compared to the example defined in the next section. 

1. The training configuration example provided below utilizes the full training set for training the model and evaluation performed on the test set, which is not the standard practise. 
2. The are differences in the current python engine compared to the version used in our DCN experiments. 

class: `TF2UpstrideLearned`
* kernel_size = `3` instead of `1` 
* kernel_initializer=`'he_normal'` instead of `'glorot_uniform'` (tensorflow default)
* kernel_regularizer= `l2(0.0005)` instead of `None`

The above class can be found in `upstride_python/upstride/generic_layers.py` of the python engine repository.

## Usage

Please make sure to follow the instructions at the root of this repository to set up everything you need to run this example.

### Train and evaluate

__Training and evalutation on CIFAR10__:

Below is the training configuration for `WSComplexNetUpstride` on CIFAR10 dataset.

```python
python train.py \
--model.name WSComplexNetUpStride \
--model.upstride_type 1 \
--model.factor 2 \
--model.input_size 32 32 3 \
--model.changing_ids beginning end_before_dense \
--model.num_classes 10 \
--model.channels_first true \
--model.conversion_params.tf2up_strategy learned \
--model.conversion_params.up2tf_strategy concat \
--dataloader.name cifar10 \
--dataloader.val_list Normalize \
--dataloader.train_split_id train \
--dataloader.val_split_id test \
--dataloader.train_list Normalize RandomHorizontalFlip Translate \
--dataloader.Normalize.only_subtract_mean true \
--dataloader.Normalize.scale_in_zero_to_one false \
--dataloader.Normalize.mean 0.491 0.482 0.446 \
--dataloader.Translate.width_shift_range 0.125 \
--dataloader.Translate.height_shift_range 0.125 \
--dataloader.batch_size 128 \
--optimizer.name sgd_nesterov \
--optimizer.lr 0.01 \
--optimizer.lr_decay_strategy.lr_params.strategy explicit_schedule \
--optimizer.lr_decay_strategy.lr_params.drop_schedule 10 100 120 150 \
--optimizer.lr_decay_strategy.lr_params.list_lr 0.1 0.01 0.001 0.0001 \
--optimizer.clipnorm 1.0 \
--optimizer.weight_decay 0.0005 \
--checkpoint_dir /tmp/DCN \
--log_dir /tmp/DCN/log/ \
--early_stopping 200 \
--num_epochs 200 \
--max_checkpoints 5
```

* `--model.name` is the name of the  model.
* `--model.upstride_type` to run the model for which upstride type.
* `--checkpoint_dir` and `--log_dir` are for storing the checkpoints and the tensorboard logs.
* `--model.input_size` input shape of the image to the model.
* `--model.changing_ids` these are indicators to automatically convert TF2Upstride and Upstride2TF. `beginning` adds the multivector component and  `end_before_dense`  converts back to real values before the logits layer.
* `--model.num_classes` number of classes present in the dataset in this case 10 classes.
* `--model.channels_first` If the user wishes to use the engine as channels first then this flag should be set `true`. Note: we would still pass input shape as (H, W, C) to the model. A Lambda layer is added in between Keras Input and TF2Upstride which would transpose the image to channels first.
* `--dataloader.name` This should `cifar10` or `cifar100`.
* `--dataloader.train_list` These are set of data augmentation that would be applied during training.
* `--dataloader.validation_list` These are set of preprocessing that would be applied during validation.
* `--dataloader.Normalize` normalizes the image and subtracts the mean from R, G, B values as specified in the arguments.
* `--dataloader.Translate` adds Random shifts in height and width during preprocessing.
* `--early_stopping` stops the training if the validation accuracy doesn't improve after the value provided. 
* `--num_epochs` number of epochs
* `--max_checkpoints` number of checkpoints to be saved to the disk before rewriting the old ones. If you need all the checkpoints ensure the value is same as number of epochs.
* `--optimizer.name` name of the optimizer
* `--optimizer.lr` initial learning rate
* `--optimizer.decay_strategy` the schedule used here is explicit with the epochs and corresponding learning rate to be changed during the training.
* `--optimizer.clipnorm` clips the gradient norm which are above 1.0
* `--optimizer.weight_decay` This is actually the L2 weight decay applied on the kernel_regularizer for the linear layers and not decay of the optimizer.
* `--model.conversion_params.tf2up_strategy` learned strategy used in DCN paper to learn the imaginary parts
* `--model.conversion_params.up2tf_strategy` concat strategy is used to stack the real and imageinary parts on the channel dimension.

__Training and evaluation on CIFAR100__:

The only change required in the above configuration is `--dataloader.name cifar100` and `--model.num_classes 100` 


### How to modify the code

All our models are availale in `/classification-api/src/models/` directory.

In order to add custom models. We only require to subclass `GenericModelBuilder` and override the `model` function.

```python
import tensorflow as tf
from .generic_model import GenericModelBuilder


class AnExampleNetwork(GenericModelBuilder)
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
# ...

  def someUserDefinedMethod(self, ...):
    pass

# ...
  def model(self, x):
    
    # model Input (handeled from generic_model.py file)

    # TF2Upstride done automatically (provided user has given model.changing_ids beginning)

    # user specific layers 
    self.layer.Conv2D(64 // self.factor)(x)

    # its possible to switch intermediate layers from hypercomplex to real and vice versa. refer to change_framework_if_necessary function. 
    # ...
    # ... 
    
    # User should not define the logits layers within the model. This will be handeled by generic_model.py file. refer def build

    # Upstride2TF done automatically (provided user has given model.changing_ids end_before_dense or end_after_dense)

```

Once the model is well defined and saved (eg example_model.py in `models` directory). Now the user would have to add the model class name `AnExampleNetwork` to the `__init__.py` file in the `classification/src/models` directory.

```python
from .example_model import AnExampleNetwork
...

model_name_to_class = {
    "AlexNet": AlexNet,
    ...
    ...
    "examplenetwork": AnExampleNetwork,
    ... 
}
```

Now in order to start the training we only need to replace the `--model.name examplenetwork`. Switching between upstride types can be done via `--model.upstride_type` argument. 

* -1 or 0 - Tensorflow
* 1 - Type 1
* 2 - Type 2
* 3 - Type 3

It's recommended the user goes through the `generic_model.py` in detail on how to use other keyword arguments.

In `train.py` file at the root of the `classification_api` folder, there are namespaces for arguments which would be useful in understanding the default values.

## References

1. Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal. “Deep Complex Networks”. In Internation Conference on Learning Representations (ICLR), 2018

[1]: https://github.com/UpStride/classification-api
[2]: http://www.cs.toronto.edu/~kriz/cifar.html
[3]: https://github.com/ChihebTrabelsi/deep_complex_networks