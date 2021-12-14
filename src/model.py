import random

import numpy as np
import tensorflow.keras as keras

from game.constants import cols, rows
from game.game import Game
from game.snake import Direction

# Input consists of the grid
INPUT_SHAPE = (rows, cols, 3)
ACTIONS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

LEARNING_RATE = 0.001
TAU = 0.125
GAMMA = 0.85

EPSILON = 0.9
EPSILON_DECAY = 1 - 0.01
FINAL_EPSILON = 0.2

BATCH_SIZE = 50


def build_model() -> keras.Model:
    model = keras.Sequential()

    model.add(
        keras.layers.Convolution2D(
            filters=32,
            kernel_size=(8, 8),
            input_shape=INPUT_SHAPE,
            name="conv1",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=(2, 2)))
    model.add(keras.layers.Convolution2D(filters=64, kernel_size=(4, 4), name="conv2"))
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=(2, 2)))
    model.add(keras.layers.Flatten(name="flatten1"))
    model.add(keras.layers.Dense(256, name="h2"))
    model.add(keras.layers.Dense(len(ACTIONS), name="out"))

    model.compile(
        loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
    )
    print(model.summary())
    return model


def get_reward(
    new_score: int,
    previous_score: int,
    is_game_over: bool,
    new_distance: float,
    old_distance: float,
) -> float:
    # Game over = -1.0
    if is_game_over:
        return -1.0
    # Goal = 1
    if new_score > previous_score:
        return 1.0

    # If we came close, then its based on how much closer / further away we got
    return 0.5 if new_distance < old_distance else -0.5


def next_action_index(model: keras.Model, state: np.ndarray, epsilon: float):
    if random.random() > epsilon:
        return random.randrange(len(ACTIONS))
    return np.argmax(model.predict(state))


class Replay:
    def __init__(self, size: int) -> None:
        self.size = size
        self.memory = []

    def send(self, item):
        return_value = None
        if len(self.memory) >= self.size:
            return_value = random.sample(self.memory, self.size)
        self.memory.append(item)
        return return_value


def reshape_input(state: np.ndarray):
    return state.reshape((1,) + INPUT_SHAPE)


def train_model(model: keras.Model):
    epochs = 10000
    game = Game()
    epsilon = EPSILON
    replay = Replay(BATCH_SIZE)
    for _ in range(epochs):
        game.reset()
        epsilon = epsilon * EPSILON_DECAY if epsilon > FINAL_EPSILON else FINAL_EPSILON
        score = game.score
        distance = game.distance()
        state = game.game_state()

        while True:
            action_index = next_action_index(model, reshape_input(state), epsilon)
            next_move = ACTIONS[action_index]

            new_state, new_score, is_game_over, new_distance = game.run_action(
                next_move
            )

            reward = get_reward(new_score, score, is_game_over, new_distance, distance)
            distance = new_distance
            score = new_score

            experience = (
                state,
                action_index,
                reward,
                reshape_input(new_state),
                is_game_over,
            )
            batch = replay.send(experience)

            state = new_state
            if batch:
                inputs = np.zeros((BATCH_SIZE,) + INPUT_SHAPE)
                targets = np.zeros((BATCH_SIZE, len(ACTIONS)))
                for i, exp in enumerate(batch):
                    exp_state, exp_move, exp_reward, exp_new_state, exp_game_over = exp
                    # State
                    inputs[i] = exp_state
                    targets[i, exp_move] = (
                        exp_reward + max(model.predict(exp_new_state)[0]) * GAMMA
                        if exp_reward >= 0 and not exp_game_over
                        else exp_reward
                    )

                model.train_on_batch(inputs, targets)

            if is_game_over:
                break

    model.save_weights("model.h5", overwrite=True)


if __name__ == "__main__":
    model = build_model()
    keras.utils.plot_model(model, show_shapes=True)
    train_model(model)
