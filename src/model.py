import tensorflow.keras as keras

from game.constants import cols, rows

# Input consists of the grid
INPUT_SHAPE = (rows, cols, 1)
ACTIONS = ["up", "right", "down", "left"]

LEARNING_RATE = 0.001


def build_model() -> keras.Model:
    model = keras.Sequential()

    model.add(
        keras.layers.Convolution2D(16, (8, 8), strides=(4, 4), input_shape=INPUT_SHAPE)
    )
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Convolution2D(32, (4, 4), strides=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dense(len(ACTIONS)))

    model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.Adam(LEARNING_RATE))

    return model


def train_model(model: keras.Model):
    epochs = 100
    for i in range(epochs):
        # TODO: Add logic for learning model
        pass


if __name__ == "__main__":
    model = build_model()

    keras.utils.plot_model(model, show_shapes=True)
