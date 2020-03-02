import numpy as np

import random

from keras.callbacks import Callback


class NeuroPlast(Callback):
    """
    Re-initialize weights that are 0 or close to 0, so that models can operate at 'full capacity'

    Arguments:
        min_value: The lowest value a weight/bias can be to be operated on
        max_value: The highest value a weight/bias can be to be operated on
        low_bound: When weights are re-initialized, this will be their lowest possible value
        high_bound: When weights are re-initialized, this will be their highest possible value
        biases: If true, NeuroPlast will also be performed on biases
    """

    def __init__(self,
                 min_value: float = -.02,
                 max_value: float = .02,
                 low_bound: float = -.05,
                 high_bound: float = .05,
                 biases: bool = False):

        super(NeuroPlast, self).__init__()

        self.min_value = min_value
        self.max_value = max_value
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.biases = biases

    def on_epoch_end(self, batch, logs=None):
        weights = self.model.get_weights()[0].copy()
        biases = self.model.get_weights()[1].copy()

        for weight in np.nditer(weights, op_flags=['readwrite']):
            if self.min_value <= weight <= self.max_value:
                weight = random.uniform(self.low_bound, self.high_bound)

        if self.biases:
            for bias in np.nditer(biases, op_flags=['readwrite']):
                if self.min_value <= weight <= self.max_value:
                    bias = random.uniform(self.low_bound, self.high_bound)

        self.model.set_weights([weights, biases])
