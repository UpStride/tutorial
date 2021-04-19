# Introduction

- [Introduction](#introduction)
    - [Pre-requisites](#pre-requisites)
    - [Example](#example)
    - [How to use UpStride](#how-to-use-upstride)
        - [TF2Upstride](#tf2upstride)
        - [Upstride2TF](#upstride2tf)

### Pre-requisites 

User should have installed the necessary dependencies mentioned in the [README.md](../README.md).

### Example

Lets look at the `train.py` file. Most of the code is standard tensorflow/keras format.

These are the arguments which are required to train a `simpleNet` architecture. All these have default values. 

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

For more information: 
```bash
usage: train.py [-h] [--upstride_type [UPSTRIDE_TYPE]] [--factor [FACTOR]] [--input_size [INPUT_SIZE ...]] [--batch_size [BATCH_SIZE]] [--num_classes [NUM_CLASSES]] [--num_epochs [NUM_EPOCHS]] [--lr [LR]]

optional arguments:
  -h, --help            show this help message and exit
  --upstride_type [UPSTRIDE_TYPE]
                        specify the upstride type [default: 1]
  --factor [FACTOR]     division factor to scale the number of channels. factor=2 means the model will have half the number of channels compare to default implementation [default: 2]
  --input_size [INPUT_SIZE ...]
                        processed shape of each image [default: [32, 32, 3]]
  --batch_size [BATCH_SIZE]
                        The size of batch per gpu [default: 16]
  --num_classes [NUM_CLASSES]
                        Number of classes [default: 10]
  --num_epochs [NUM_EPOCHS]
                        The number of epochs to run [default: 10]
  --lr [LR]             initial learning rate [default: 0.001]
```

### How to use UpStride

UpStride engine provides layers that transforms real valued tensor into the following representations.

- type1 - Complex 
- type2 - Quaternion
- type3 - bi-Quaternion (experimental)

The main module, where the UpStride custom layers are implemented.

UpStride's engine is divided in three main modules `type1`, `type2` and `type3`. Every module has the same type of layers and functions. They be imported in the following way:

```python
# type 1
from upstride.type1.tf.keras import layers
# type 2
from upstride.type2.tf.keras import layers
# type 3
from upstride.type3.tf.keras import layers
```

To use it, start by importing the layers package from the upstride type you want to use. 

Then the neural network can be define using keras API. Start by defining a Input, convert it to Upstride by calling `layers.TF2Upstride` and build
the neural network the same way you do with Keras. At the end of the neural network call `layers.Upstride2TF` to convert back upstride tensor to TensorFlow tensor.

For training and inference, all TensorFlow tools can be used (distribute strategies, mixed-precision training...)

Only Python 3.6 or later is supported.

In the below example, we have a simple network architecture.

* The upstride layers are imported by passing `upstride_type` to the python function `load_upstridetype`. So user is required to pass `--upstride_type <<value>>` to the python file.
 
* We have defined `layers.TF2Upstride(strategy="basic")`. The output of this layer would be Real and imaginary parts initialized with zeros.

* The intermediate blocks are omitted here for the sake of simplicity. You may refer to the `simpleNet` definition in the `train.py` file. 

* Just before softmax activation we convert back to real `layers.Upstride2TF(strategy="basic")` as the output are probablities.



```python
def simpleNet(input_size: List[int], factor: int, num_classes: int) -> tf.keras.Model:

    # import the respective upstride type layers
    layers = load_upstridetype(args['upstride_type'])

    inputs = tf.keras.layers.Input(shape=input_size)

    # TF to Upstride
    x = layers.TF2Upstride(strategy="basic")(inputs)

    # refer to the simpleNet definition in train.py
    # Block 1
    x = layers.Conv2D(32 // factor, (3, 3))(x)
    #...
    #...

    # Upstride to TF
    x = layers.Upstride2TF(strategy="basic")(x)
    outputs = tf.keras.layers.Activation("softmax")(x)

    model = tf.keras.models.Model(inputs, outputs)

    # ...

    return model
```

The simplest way to adapt your own code is described above: you only need to call the UpStride layers instead of regular TensorFlow. It allows the user to readily benefit from our Engine. This is the "vanilla" way of using UpStride.

However, due to the way the python engine is implemented, the vanilla approach results in a model that contains more free parameters than its pure TensorFlow counterpart. 

The `factor` is passed to reduce the number of channels at each given block. This ensures the number of free paramters are comparable with real valued networks. 

* type 1 - factor 2
* type 2 - factor 4
* type 3 - factor 8

##### TF2Upstride

Several strategies are available to convert TensorFlow tensors to Upstride tensors.

`basic` or ` `: imaginary components are initialized with zeros.

```python
x = layers.TF2Upstride()(inputs)
```
`learned`: a small neural network (ResNet blocks) (2 convolutions with 3x3 kernel and 3 channels) is used to learn the imaginary components.

```python
x = layers.TF2Upstride(strategy='learned')(inputs)
```

##### Upstride2TF

Four strategies are possible:

- `basic` outputs a tensor that keeps only the real values of the tensor. 
- `concat` generates a vector by concatinating the imaginary components on the final dimension.
- `max_pool` (experimental) ouputs a tensor that takes the maximum values across the real and imaginary parts.
- `avg_pool` (experimental) ouputs a tensor that takes average across the real and imaginary parts.

e.g:
```python
x = layers.Upstride2TF(strategy="concat")(x)
```

Finally, We can now build, complile and train the model. Refer to the `train.py` for the code. 
