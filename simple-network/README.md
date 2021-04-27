# Simple Network

This is a basic example on how to train a simple convolutional neural network (CNN) using the upstride engine. We also provide guidance on how to easily convert your own neural networks (NNs) from real to hyper-complex using the upstride layers.

# Table of Contents
- [Simple Network](#simple-network)
- [Table of Contents](#table-of-contents)
- [Description](#description)
  - [Pre-requisites](#pre-requisites)
  - [Example](#example)
  - [Usage](#usage)
# Description

In this example, we show how to train and evaluate a simple CNN on the CIFAR-10 dataset. The network is composed of two convolutional layers followed by max pooling and two fully connected layers. The goal is to show how it is possible to create a NN that use different underlying algebras using the upstride engine. Before proceeding with this example, we recommend reading the documentation of the upstride engine.

## Pre-requisites

User should have installed the necessary dependencies mentioned in the [README.md](../README.md).

## Example

Run:

```bash
python train.py \
    --upstride_type 1 \
    --factor 2 \
    --input_size 32 32 3 \
    --batch_size 16 \
    --num_classes 10 \
    --num_epochs 10 \
    --lr 0.001
```

For a description of all the available arguments, run `python train.py -h` .

## Usage

The upstride engine provides layers that allow you to define networks that work with:

* real numbers (type0) - used for compatibility of models with layers like TF2Upstride and Upstride2TF with real number networks
* complex numbers (type1)
* quaternions (type2)

The upstride engine re-implementes the main tensorflow layers for each of the three types described above. To use the layers of each type, simply import them as:

```python
# type 0
from upstride.type0.tf.keras import layers
# type 1
from upstride.type1.tf.keras import layers
# type 2
from upstride.type2.tf.keras import layers
```

The upstride layers maintain the original tensorflow/keras API, making it very straightforward to use. Let's have a look at the network definition:

```python
def simpleNet(input_size: List[int], factor: int, num_classes: int) -> tf.keras.Model:

    # import the respective upstride type layers
    layers = load_upstridetype(args['upstride_type'])

    inputs = tf.keras.layers.Input(shape=input_size)
    # TF to Upstride
    x = layers.TF2Upstride()(inputs)

    # Block 1
    x = layers.Conv2D(32 // factor, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3,3), strides=(1,1))(x)

    # Block 2
    x = layers.Conv2D(64 // factor, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2))(x)

    # Block 3
    x = layers.Flatten()(x)
    x = layers.Dense(128 // factor)(x)

    # Logits
    x = layers.Dense(num_classes, use_bias=True)(x)

    # Upstride to TF
    x = layers.Upstride2TF(strategy="basic")(x)
    outputs = tf.keras.layers.Activation("softmax")(x)

    model = tf.keras.models.Model(inputs, outputs)

    # print the model summary
    model.summary()

    return model
```

We can see that the code looks almost identical to a standard network definition in tensorflow, except from:

- `layers.TF2Upstride`, used to convert the input tensor to the chosen `upstride_type`, and
- `layers.Upstride2TF`, used to convert back to the standard tensorflow tensor.

For example, if we want to define a complex-valued NN, we can pass the argument `--upstride_type 1`. The `TF2Upstride` layer will then convert the real-valued tensors in input to complex-valued tensors, and `Upstride2TF` will convert the complex-valued tensors back to real-valued tensors for the output. The upstride engine supports different strategies for both `TF2Upstride` and `Upstride2TF`. Please refer to the engine documentation for the descriptions of these strategies.