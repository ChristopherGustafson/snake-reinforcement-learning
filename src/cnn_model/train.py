import random
from datetime import datetime

import numpy as np
import tensorflow.keras as keras
from tqdm import tqdm

from cnn_model.constants import (
    BATCH_SIZE,
    EPSILON,
    EPSILON_DECAY,
    FEATURES,
    FINAL_EPSILON,
    GAMMA,
    MEMORY_SIZE,
    WEIGHTS_SAVE_FILE_NAME,
)
from cnn_model.replay import Replay
from cnn_model.utils import get_initial_state, get_reward, get_target
from game.constants import COLS, ROWS
from game.game import Game, Model
from game.snake import Direction


def train_model(
    model: keras.Model,
    epochs: int,
    store_weights: bool,
    use_graphics: bool,
    start_epoch: int = 0,
):
    epsilon = EPSILON
    epoch = start_epoch
    game = Game(Model.CNN, use_graphics)
    replay = Replay(MEMORY_SIZE, GAMMA)
    progress_bar = tqdm(total=epochs)
    progress_bar.update(start_epoch)
    while epoch < epochs:
        game_over = False
        current_state = next_state = get_initial_state(game)
        current_score = 0
        current_distance = game.distance()
        while not game_over:
            predictions = model.predict(current_state)[0]
            if random.random() < epsilon:
                action_index = random.randrange(len(Direction))
            else:
                action_index = np.argmax(predictions)
            action = Direction(action_index)
            # Execute actoin
            state, new_score, game_over, new_distance = game.run_action(action)

            # Get the reward
            reward = get_reward(
                current_distance, new_distance, current_score, new_score, game_over
            )
            current_distance = new_distance
            current_score = new_score

            # Reshape state
            state = np.reshape(state, (1, ROWS, COLS, 1, FEATURES))
            # Add new game frame to the next state, delete last (oldest) state
            next_state = np.append(next_state, state, axis=3)
            next_state = np.delete(next_state, 0, axis=3)

            experience = [current_state, action_index, reward, next_state]

            # Train the network on the current step taken, for 1 epoch
            targets = np.array([get_target(model, GAMMA, experience, game_over)])
            model.fit([current_state], targets, epochs=1, verbose=False)

            # Remember the transition and train the neural network on batch
            replay.remember(experience, game_over)
            inputs, targets = replay.get_batch(model, BATCH_SIZE)
            model.train_on_batch(inputs, targets)

            current_state = next_state

        if epsilon > FINAL_EPSILON:
            epsilon -= EPSILON_DECAY

        if store_weights and epoch % 100 == 0 and (epoch - start_epoch) > 10:
            print("Storing weights...")
            model.save_weights(
                WEIGHTS_SAVE_FILE_NAME.format(
                    ROWS, COLS, epoch, datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
                )
            )

        epoch += 1
        progress_bar.update()

    print("Storing weights...")
    model.save_weights(
        WEIGHTS_SAVE_FILE_NAME.format(
            ROWS, COLS, epochs, datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        )
    )
