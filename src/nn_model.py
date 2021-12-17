import collections
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from game.game import Game, Player
from game.snake import Direction

ACTIONS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

# Agent constants
HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = 150
STATE_SPACE_SIZE = 12
ACTION_SPACE = 4
LEARNING_RATE = 0.0001
EPSILON = 0.95
EPSILON_DECAY = 0.9995
FINAL_EPSILON = 0.05
MAX_MEMORY_SIZE = 3000
GAMMA = 0.85
BATCH_SIZE = 200

MODEL_SAVE_INTERVAL = 100
TOTAL_GAMES = 4000
EPOCHS_PER_GAME = 10


class DQN_Agent:
    hidden_layers = HIDDEN_LAYERS
    hidden_layer_size = HIDDEN_LAYER_SIZE
    state_space_size = STATE_SPACE_SIZE
    action_space_size = ACTION_SPACE

    def __init__(
        self,
        epsilon: float = EPSILON,
        learning_rate: float = LEARNING_RATE,
        max_memory_size: int = MAX_MEMORY_SIZE,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.state_memory = collections.deque(maxlen=max_memory_size)
        self.reward_memory = collections.deque(maxlen=max_memory_size)
        self.target_memory = collections.deque(maxlen=max_memory_size)

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

    def replay_memory(self, epochs_per_game):
        state_batch = np.squeeze(self.state_memory)
        # Current state batch = all stored previous states except latest
        current_state_batch = state_batch[0:-1, :]
        # Next state batch = all stored previous states except oldest
        next_state_batch = state_batch[1:, :]

        # Previous rewards, except oldest
        reward_batch = np.array(self.reward_memory)
        reward_batch = reward_batch[1:]

        for _ in range(epochs_per_game):
            # Predict target based on current state batch
            target_batch = self.model.predict(current_state_batch)
            # Find the actions for the highest prediction
            action_i = np.argmax(target_batch, axis=1)
            # Update target with prediction from new batch
            target_batch[
                np.arange(target_batch.shape[0]), action_i
            ] = reward_batch + GAMMA * np.max(
                self.model.predict(next_state_batch), axis=1
            )

            # fit the model, using the current state batch and the updated target_batch
            dataset = tf.data.Dataset.from_tensor_slices(
                (current_state_batch, target_batch)
            )
            dataset = (
                dataset.shuffle(BATCH_SIZE)
                .prefetch(buffer_size=BATCH_SIZE)
                .batch(BATCH_SIZE)
            )

            self.model.fit(dataset, verbose=False, epochs=1)

    def train_agent(self, games, epochs_per_game):
        game = Game(Player.NN)
        highscore = 0
        total_scores = 0
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
            self.replay_memory(epochs_per_game)

            total_scores = total_scores + score
            if score > highscore:
                highscore = score
            if game_i % MODEL_SAVE_INTERVAL == 0:
                self.model.save_weights("nn_model.h5", overwrite=True)
                avg_score = 0
                if game_i != 0:
                    avg_score = total_scores / MODEL_SAVE_INTERVAL
                    total_scores = 0
                print(
                    f"Finished {game_i} games: , epsilon: {self.epsilon}, avg score: {avg_score}, highscore: {highscore}"
                )


if __name__ == "__main__":
    agent = DQN_Agent()
    agent.train_agent(TOTAL_GAMES, EPOCHS_PER_GAME)
