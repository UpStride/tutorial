# Introduction

## Index

1. Build the docker file.
2. launch the docker.
3. Imports, arguments and loading the dataset.
4. Model Definition.
5. Build and compile the model.
6. Training and Evaluation.

### Build the docker file. 

Let's prepare the environment by building the docker file. 
The `upstride.dockerifle` contains the UpStride python engine and required dependencies to be installed.

We use the `makefile` to build and run the dockers

From the root of this directory type in the shell:
> make build_upstride

you would see the docker being pulled from our cloud registry, dependencies being installed and finally the docker image.

### Launch the docker

From the shell run the below command:
> make run_upstride

The docker is launched and you should see the docker terminal in the same window. Note: You would be logged in as root. 

If the user wishes to use a specific dataset, Ensure to mount them via `-v local_path:path_inside_the_docker` 

```make
run_upstride:
    @docker run -it --rm --gpus all --privileged \
        -v $$(pwd):/opt \
        -v /path_to_upstride_python_engine_repo/:/upstride_python \ # if users wishes to add custom layer. requires a pip install or add upstride folder to python path
        -v ~/path_to_your_dataset/:/path_to_your_dataset \ # path to users dataset
        upstride/classification_api:upstride-1.0 \
        bash
```
### Imports, arguments and loading the dataset

* Import the necessary libraries.
* arguments is a better argument parser developed by UpStride.
* load the CIFAR10 dataset and normalize the x_train and x_test 

```python
from typing import List

import tensorflow as tf 

from upstride.type1.tf.keras import layers
import upstride_argparse as argparse

arguments = [
    #   [type, arg name, default, str definition, validation(optional)]
        [int, "factor", 2, 'division factor to scale the number of channel. factor=2 means the model will have half the number of channels compare to default implementation'],
        ['list[int]', "input_size", [32, 32, 3], 'processed shape of each image'],
        [int, 'batch_size', 16, 'The size of batch per gpu', lambda x: x > 0],
        [int, "num_classes", 10, 'Number of classes', lambda x: x > 0],
        [int, 'num_epochs', 10, 'The number of epochs to run', lambda x: x > 0],
        [float, "lr", 0.001, 'initial learning rate', lambda x: x > 0],
    ]

# download CIFAR10 dataset
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# normalize the x_train and x_test
x_train, x_test = x_train / 255.0, x_test / 255.0 

# argparse utility returns python dict 
args = argparse.parse_cmd(arguments)
```

### Model Definition

In the below example, we have a simple network architecture similar to AlexNet.

* We have defined `layers.TF2Upstride(strategy="basic")`. The output of this layer would be Real and imaginary parts initialized with zeros.

* Then we have Layer 1 till Layer 5 which are the standard way. 

* Just before softmax activation we convert back to real as the output are probablities.


```python
def simpleNet(input_size: List[int], factor: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(input_size)
    # TF to UpStride
    x = layers.TF2Upstride(strategy="basic")(inputs) # basic or " " 
    # Layer 1
    x = layers.Conv2D(128//factor, (3,3), 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3,3), strides=(1,1))(x)
    # Layer 2
    x = layers.Conv2D(256//factor, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2))(x)
    # Layer 3
    x = layers.Conv2D(256//factor, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2))(x)
    # Layer 4
    x = tf.keras.layers.Flatten()(x)
    x = layers.Dense(384//factor, activation="relu")(x)
    # Layer 5
    x = layers.Dense(num_classes, use_bias=True)(x)

    # Upstride to TF
    x = layers.Upstride2TF(strategy="basic")(x)
    outputs = tf.keras.layers.Activation("softmax")(x)

    model = tf.keras.models.Model(inputs, outputs)

    # print the model summary
    model.summary()

    return model
```

Note: The imaginary parts are stacked along the Batch hence the `model.summary()` looks cleaner.

### Build and compile the model

* we pass the necessary arguments to the `simpleNet` model and build it.
* Then we complie the model before we train.

```python
# Build the model
model = simpleNet(args['input_size'], args['factor'], args['num_classes'])

# Complie the model 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args['lr']),
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
```

### Training and Evaluation

* We train the model with 90% of the train set and evaluate on the remaining 10%.
* Finally, we evaluate the test set on the trained model.

```python
# Training 
model.fit(x_train,y_train,
          batch_size=args['batch_size'],
          epochs=args['num_epochs'],
          shuffle=True,
          validation_split=0.1)

# Evaluation
model.evaluate(x_test,y_test,batch_size=args['batch_size'])
```
