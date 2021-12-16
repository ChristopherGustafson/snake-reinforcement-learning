import collections
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from game.game import Game
from game.snake import Direction

ACTIONS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]


# Constants, to be add in constants.py

HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = 150
STATE_SPACE_SIZE = 12
ACTION_SPACE = 4
LEARNING_RATE = 0.0005
EPSILON = 0.9
EPSILON_DECAY = 0.9995
FINAL_EPSILON = 0.0
MAX_MEMORY_SIZE = 3000
GAMMA = 0.9
BATCH_SIZE = 200


class DQN_Agent:
    def __init__(self):
        self.hidden_layers = HIDDEN_LAYERS
        self.hidden_layer_size = HIDDEN_LAYER_SIZE
        self.state_space_size = STATE_SPACE_SIZE
        self.action_space_size = ACTION_SPACE
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON

        self.state_memory = collections.deque(maxlen=MAX_MEMORY_SIZE)
        self.reward_memory = collections.deque(maxlen=MAX_MEMORY_SIZE)
        self.target_memory = collections.deque(maxlen=MAX_MEMORY_SIZE)
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()

        # Input layer takes the number of features in the state space as input
        model.add(keras.layers.Input(self.state_space_size))

        # Add set of hidden layers, using ReLU activation function
        for _ in range(self.hidden_layers):
            model.add(keras.layers.Dense(self.hidden_layer_size, activation="relu"))

        # Output layer results in the number of output as the possible actions
        # as it should tell us what action to take in the next step.
        model.add(keras.layers.Dense(self.action_space_size, activation="linear"))

        # Use mean square error loss function and adam optimizer
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        return model

    def remember(self, state, reward):
        self.state_memory.append(state)
        self.reward_memory.append(reward)

        # Compute target if we have seen two previous states
        if len(self.state_memory) > 1:
            prediction_last_state = self.model.predict(self.state_memory[-2])
            prediction_curr_state = self.model.predict(state)
            # Compute target according to q-learning equation
            target = reward + GAMMA * np.max(prediction_curr_state)
            global_target = prediction_last_state
            global_target[0, np.argmax(prediction_last_state)] = target
            # Add to memory
            self.target_memory.append(global_target)

    def get_reward(
        self, new_score: int, previous_score: int, is_game_over: bool
    ) -> float:
        # Game over = -20
        if is_game_over:
            return -20.0
        # Eatenn apple = 10
        if new_score > previous_score:
            return 10.0
        # If not hit anything = 0
        return 0.0

    # Decrease epsilon with a factor of EPSILON_DECAY, until it reaches FINAL_EPSILON
    def epsilon_cool_down(self):
        self.epsilon = (
            self.epsilon * EPSILON_DECAY
            if self.epsilon > FINAL_EPSILON
            else FINAL_EPSILON
        )

    def next_action(self, state):
        # Epsilon shrinks over time, if random < epsilon, take the
        # next move according to the model prediction
        if random.random() > self.epsilon:
            action_predictions = self.model.predict(state)
            max_action_i = np.argmax(action_predictions)
            return ACTIONS[max_action_i]
        # Else, make a random action, will be more common in the beginning
        # of the rounds
        random_action_i = random.randrange(len(ACTIONS))
        # Decrease epsilon to make it more unlikely next time to take random action
        self.epsilon_cool_down()
        return ACTIONS[random_action_i]

    def reshape_input(self, state: np.ndarray):
        return state.reshape((1, self.state_space_size))

    def train_agent(self, games, epochs_per_game):
        game = Game()
        highscore = 0
        for game_i in range(games):
            # Initial game state
            game.reset()
            score = game.score
            state = game.game_state()

            # Play a game and learn from it
            game_over = False
            while not game_over:
                # Make game move according to next_action function
                next_action = self.next_action(self.reshape_input(state))
                new_state, new_score, is_game_over, _ = game.run_action(next_action)
                # Figure out the reward for the result of the chose action
                reward = self.get_reward(new_score, score, is_game_over)
                # Save state, reward to memory and calculate target
                self.remember(self.reshape_input(new_state), reward)

                # Short term model fitting after every move
                if len(self.state_memory) > 1:
                    self.model.fit(
                        self.state_memory[-2],
                        self.target_memory[-1],
                        verbose=False,
                        epochs=1,
                    )

                # Save new state for next iteration
                state = new_state
                score = new_score
                game_over = is_game_over

            # After game is finished, train the model on random samples for
            # specified number of epochs
            state_batch = np.squeeze(self.state_memory)
            current_state_batch = state_batch[0:-1, :]
            next_state_batch = state_batch[1:, :]

            reward_batch = np.array(self.reward_memory)
            reward_batch = reward_batch[1:]

            for _ in range(epochs_per_game):
                target_batch = self.model.predict(current_state_batch)
                action_i = np.argmax(target_batch, axis=1)
                target_batch[
                    np.arange(target_batch.shape[0]), action_i
                ] = reward_batch + GAMMA * np.max(
                    self.model.predict(next_state_batch), axis=1
                )

                dataset = tf.data.Dataset.from_tensor_slices(
                    (current_state_batch, target_batch)
                )
                dataset = (
                    dataset.shuffle(200)
                    .prefetch(buffer_size=BATCH_SIZE)
                    .batch(BATCH_SIZE)
                )

                self.model.fit(dataset, verbose=False, epochs=1)
            if score > highscore:
                highscore = score
            print(
                f"Finished game: {game_i}, epsilon: {self.epsilon}, score: {score}, highscore: {highscore}"
            )
        self.model.save_weights("model.h5", overwrite=True)


if __name__ == "__main__":
    agent = DQN_Agent()
    games = 4000
    epochs_per_game = 10
    agent.train_agent(games, epochs_per_game)
