from typing import List

import tensorflow as tf 

import upstride_argparse as argparse

arguments = [
        [int, "upstride_type", 1, 'specify the upstride type', lambda x: x > 0 and x < 4],
        [int, "factor", 2, 'division factor to scale the number of channels. factor=2 means the model will have half the number of channels compare to default implementation'],
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

def load_upstridetype(upstride_type):
    if upstride_type == 1:
        from upstride.type1.tf.keras import layers
        return layers
    elif upstride_type == 2:
        from upstride.type2.tf.keras import layers
        return layers
    elif upstride_type == 3:
        # FIXME type 3 missing aren't we including this as experimental?
        from upstride.type3.tf.keras import layers
        return layers
    else:
        raise ValueError(f"Upstride_type {upstride_type} not a valid type")

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
