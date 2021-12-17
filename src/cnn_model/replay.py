import random

import numpy as np
import tensorflow.keras as keras

from cnn_model.constants import BATCH_SIZE, GAMMA, MEMORY_SIZE
from cnn_model.utils import get_target


class Replay:
    def __init__(self, max_memory: int = MEMORY_SIZE, discount: float = GAMMA) -> None:
        """
        Initialize the replay memory

        :param max_memory: The size of the memory
        :param discount: Discount factor, used when calculating the target prediction
        """
        self.max_memory = max_memory
        self.experience_memory = []
        self.discount = discount

    def remember(self, transition, game_over):
        """
        Remember the transition (experience), and whether the transition resulted in game over.
        Remove the oldest transition of we get above the memory limit.
        """
        self.experience_memory.append([transition, game_over])
        if len(self.experience_memory) > self.max_memory:
            self.experience_memory.pop(0)

    def get_batch(self, model: keras.Model, batch_size: int = BATCH_SIZE):
        """
        Returns a random sample of training data, based on previous experience

        :param model: The model we want to train
        :param batch_size: How large batch should (preferebly) be
        """

        # If our memory isn't large enough yet, just take the whole memory instead
        real_batch_size = min(batch_size, len(self.experience_memory))

        inputs = np.zeros(shape=(real_batch_size,) + model.input_shape[1:])
        targets = np.zeros(shape=(real_batch_size, model.output_shape[-1]))

        samples = random.sample(self.experience_memory, real_batch_size)

        for i, (experience, game_over) in enumerate(samples):
            inputs[i] = experience[0]
            # Get the current predictions for the state
            targets[i] = get_target(model, self.discount, experience, game_over)

        return inputs, targets
