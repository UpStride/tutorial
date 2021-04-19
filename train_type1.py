from typing import List

import tensorflow as tf 

from upstride.type1.tf.keras import layers
import upstride_argparse as argparse

arguments = [
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

# A simple AlexNet like architecture
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

# Build the model
model = simpleNet(args['input_size'], args['factor'], args['num_classes'])

# Complie the model 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args['lr']),
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

# Training 
model.fit(x_train,y_train,
          batch_size=args['batch_size'],
          epochs=args['num_epochs'],
          shuffle=True,
          validation_split=0.1)

# Evaluation
model.evaluate(x_test,y_test,batch_size=args['batch_size'])
