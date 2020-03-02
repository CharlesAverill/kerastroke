import numpy as np

from keras.callbacks import Callback


class Pruning(Callback):
    """
    Set weights that fall between a certain bound to a certain value. Typically used to reduce model size by setting
    weights that are close to 0.0, to 0.0.

    Arguments:
        value: The value that pruned weights will be set to. (this should usually be 0.0)
        min_value: The lowest value a weight can be to be oeprated on
        max_value: The highest value a weight can be to be operated on
        cutoff: The number of epochs that Pruning will be performed
        biases: If true, Pruning will also be performed on biases
    """

    def __init__(self,
                 value: float = 0.0,
                 min_value: float = -.2,
                 max_value: float = .2,
                 cutoff: int = -1,
                 biases: bool = False):

        super(Pruning, self).__init__()

        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.cutoff = cutoff
        self.biases = biases

        self.epochs_completed = 0

    def on_epoch_end(self, batch, logs=None):
        if self.cutoff > self.epochs_completed or self.cutoff == -1:
            weights = self.model.get_weights()[0].copy()
            biases = self.model.get_weights()[1].copy()

            for weight in np.nditer(weights, op_flags=['readwrite']):
                if self.min_value <= weight <= self.max_value:
                    weight = self.value

            if self.biases:
                for bias in np.nditer(biases, op_flags=['readwrite']):
                    if self.min_value <= bias <= self.max_value:
                        bias = self.value

            self.model.set_weights([weights, biases])
