import numpy as np
import tensorflow.keras as keras

from game.constants import cols, rows
from game.game import Game
from game.snake import Direction

# Input consists of the grid
INPUT_SHAPE = (rows, cols, 1)
ACTIONS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

LEARNING_RATE = 0.001


def build_model() -> keras.Model:
    model = keras.Sequential()

    model.add(
        keras.layers.Convolution2D(
            kernel_size=5,
            filters=32,
            strides=2,
            input_shape=INPUT_SHAPE,
            padding="SAME",
            name="conv1",
        )
    )
    model.add(keras.layers.Activation("relu"))
    model.add(
        keras.layers.Convolution2D(
            filters=64, kernel_size=5, strides=2, padding="SAME", name="conv2"
        )
    )
    model.add(keras.layers.Flatten(name="flatten1"))
    model.add(keras.layers.Dense(1024, activation="relu", name="h2"))
    model.add(keras.layers.Dense(len(ACTIONS), name="out"))

    model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.Adam(LEARNING_RATE))
    print(model.summary())
    return model


def train_model(model: keras.Model):
    epochs = 1
    for _ in range(epochs):
        game = Game()

        state = game.game_state()
        while True:
            # Reshape the input to be accepted by keras
            keras_input = state.reshape((1,) + INPUT_SHAPE)
            prediction = model.predict(keras_input)
            next_move_index = np.argmax(prediction[0])
            next_move = ACTIONS[next_move_index]
            state, score, is_game_over = game.run_action(next_move)
            if is_game_over:
                break


if __name__ == "__main__":
    model = build_model()
    keras.utils.plot_model(model, show_shapes=True)
    train_model(model)
