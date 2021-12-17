import random
from datetime import datetime

import numpy as np
import tensorflow.keras as keras
from tqdm import tqdm

from cnn_model.constants import (
    BATCH_SIZE,
    EPSILON,
    EPSILON_DECAY,
    FINAL_EPSILON,
    GAMMA,
    MEMORY_SIZE,
    WEIGHTS_SAVE_FILE_NAME,
)
from cnn_model.replay import Replay
from cnn_model.utils import add_new_state, get_initial_state, get_reward, get_target
from game.constants import COLS, ROWS
from game.game import Game, Player
from game.snake import Direction


def train_model(
    model: keras.Model,
    epochs: int,
    store_weights: bool,
    use_graphics: bool,
    start_epoch: int = 0,
):
    """
    Train the CNN model

    :param model: The model to train
    :param epochs: For how many epochs (games) we should run
    :param store_weights: Whether or not to store the weights from training
    :param use_graphics: Whether or not to display the game on screen
    :param start_epoch: Offset for when the initial epoch is started. Useful when re-training models.
    """
    epsilon = EPSILON
    epoch = start_epoch
    # Initialize the game
    game = Game(Player.CNN, use_graphics)
    # Initialize the memory
    replay = Replay(MEMORY_SIZE, GAMMA)

    progress_bar = tqdm(total=epochs)
    progress_bar.update(start_epoch)

    while epoch < epochs:
        game_over = False
        # New game, get the initial state
        current_state = next_state = get_initial_state(game)
        current_score = 0
        current_distance = game.distance()
        while not game_over:
            # Make a prediction based on the current state
            predictions = model.predict(current_state)[0]
            # Epsilon shrinks over time, if random < epsilon, take the prediction from the model
            if random.random() < epsilon:
                action_index = random.randrange(len(Direction))
                action = Direction(action_index)
            else:
                action_index = np.argmax(predictions)

            # Execute the action
            action = Direction(action_index)
            state, new_score, game_over, new_distance = game.run_action(action)

            # Get the reward
            reward = get_reward(
                current_distance, new_distance, current_score, new_score, game_over
            )
            current_distance = new_distance
            current_score = new_score

            # Reshape state
            next_state = add_new_state(state, next_state)

            # Packet our trainsition as an experience
            experience = [current_state, action_index, reward, next_state]

            # Train the network on the current step taken, for 1 epoch
            targets = np.array([get_target(model, GAMMA, experience, game_over)])
            model.fit([current_state], targets, epochs=1, verbose=False)

            # Remember the transition and train the neural network on a random mini-batch
            replay.remember(experience, game_over)
            inputs, targets = replay.get_batch(model, BATCH_SIZE)
            model.train_on_batch(inputs, targets)

            current_state = next_state

        # Decrease epsilon based on decay rae
        if epsilon > FINAL_EPSILON:
            epsilon -= EPSILON_DECAY

        # Store the weights every 100'th epoch
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
