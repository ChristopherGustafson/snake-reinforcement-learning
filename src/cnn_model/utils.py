import numpy as np
import tensorflow.keras as keras

from cnn_model.constants import (
    FEATURES,
    REWARD_CLOSER,
    REWARD_DIE,
    REWARD_LIVING,
    REWARD_WIN,
    STORED_LAST_STATES,
)
from game.constants import COLS, ROWS
from game.game import Game
from game.snake import Direction


def get_target(model: keras.Model, gamma, experience, game_over):
    current_state, action, reward, next_state = experience
    # Predict the current state (Q values)
    target = model.predict(current_state)[0]

    # The Q value of next state
    Q_sa = np.max(model.predict(next_state)[0])
    # For the action we took, replace the current Q value
    target[action] = reward if game_over else reward + gamma * Q_sa
    return target


def get_initial_state(game: Game):
    state = np.zeros((1, ROWS, COLS, STORED_LAST_STATES, FEATURES))

    for i in range(STORED_LAST_STATES):
        state[:, :, :, i] = game.reset()

    return state


def get_reward(
    previous_distance: float,
    new_distance: float,
    previous_score: int,
    new_score: int,
    game_over: bool,
):
    if game_over:
        return REWARD_DIE

    if new_score > previous_score:
        return REWARD_WIN

    if new_distance < previous_distance:
        return REWARD_CLOSER

    return REWARD_LIVING


def add_new_state(new_state, current_state):
    # Reshape state
    new_state_reshaped = np.reshape(new_state, (1, ROWS, COLS, 1, FEATURES))
    # Add new game frame to the next state, delete last (oldest) state
    new_current_state = np.append(current_state, new_state_reshaped, axis=3)
    new_current_state = np.delete(new_current_state, 0, axis=3)
    return new_current_state


def predict_move(cnn_model, state):
    predictions = cnn_model.predict(state)[0]  # type: ignore
    action_index = np.argmax(predictions)
    return Direction(action_index)
