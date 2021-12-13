import tensorflow.keras as keras

from game.constants import cols, rows


def build_model() -> keras.Model:

    # Input consists of the grid size, snakes position and the goal
    grid_input = keras.layers.Input(shape=(rows, cols))

    output = keras.layers.Dense(4, activation=keras.activations.softmax, name="out")(
        grid_input
    )

    model = keras.models.Model(inputs=[grid_input], outputs=output)

    model.compile(
        loss=keras.losses.MSE,
    )

    return model


if __name__ == "__main__":
    model = build_model()

    keras.utils.plot_model(model, show_shapes=True)
