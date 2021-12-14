import tensorflow.keras as keras

from cnn_model.constants import FEATURES, LEARNING_RATE, STORED_LAST_STATES
from game.constants import COLS, ROWS
from game.snake import Direction


def build_model() -> keras.Model:
    # The input is: The grid, N last states, and some features (fruit and snake)
    game_input = keras.layers.Input(
        shape=(ROWS, COLS, STORED_LAST_STATES, FEATURES),
    )

    conv1 = keras.layers.Convolution3D(
        filters=32,
        kernel_size=(3, 3, 2),
        activation="relu",
        name="conv1",
    )(game_input)

    pooling1 = (
        keras.layers.MaxPooling3D(
            pool_size=(2, 2, 1),
        )
    )(conv1)

    conv2 = (
        keras.layers.Convolution3D(
            filters=64,
            kernel_size=(2, 2, 1),
            activation="relu",
            name="conv2",
        )
    )(pooling1)

    flattened = keras.layers.Flatten(
        name="flatten1",
    )(conv2)

    h1 = keras.layers.Dense(
        256,
        activation="relu",
        name="h1",
    )(flattened)

    output = keras.layers.Dense(
        len(Direction),
        activation=keras.activations.linear,
        name="out",
    )(h1)

    model = keras.Model(inputs=[game_input], outputs=output)

    model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
    )
    return model
