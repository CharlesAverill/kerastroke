import random
import numpy as np

from tensorflow.keras.callbacks import Callback


class Stroke(Callback):
    """
    Re-initialize a percentage of weights in a layer or model at the end of every epoch to improve generalization.
    """

    def __init__(self,
                 set_value: float = None,
                 low_bound: float = -.05,
                 high_bound: float = .05,
                 volatility_ratio: float = .05,
                 cutoff: int = -1,
                 decay: float = None,
                 do_weights: bool = True,
                 do_biases: bool = False):
        """
        :param set_value: re-initialized weights will be set to this value, rather than a random one
        :param low_bound: low bound for weight re-initialization
        :param high_bound: high bound for weight re-initialization
        :param volatility_ratio: percentage of weights to be re-initialized
        :param cutoff: number of epochs to perform PBWOs
        :param decay: Every epoch, v_ratio is multiplied by this number. decay can be greater than 1.0,
                      but v_ratio will never exceed 1.0
        :param do_weights: perform stroke on weights
        :param do_biases: perform stroke on biases
        """

        super(Stroke, self).__init__()

        self.value = set_value
        self.low_bound = low_bound
        self.high_bound = high_bound
        assert volatility_ratio < 1.0, "volatility_ratio must be less than 1.0"
        self.v_ratio = volatility_ratio
        self.cutoff = cutoff
        self.decay = decay
        self.do_weights = do_weights
        self.do_biases = do_biases

        self.epochs_completed = 0

    def __stroke(self, matrix):
        """
        :param matrix: np_array to perform stroke on
        :return: modified matrix
        """

        matrix_shape = matrix.shape
        flat_matrix = matrix.flatten()
        ls_matrix = flat_matrix.tolist()

        num_items = len(ls_matrix)

        for stricken in range(int(num_items * self.v_ratio)):
            if self.value:
                ls_matrix[random.randint(0, len(ls_matrix) - 1)] = self.value
            else:
                ls_matrix[random.randint(0, len(ls_matrix) - 1)] = random.uniform(self.low_bound, self.high_bound)

        matrix = np.array(ls_matrix)
        matrix = np.reshape(matrix, matrix_shape)

        return matrix

    def on_epoch_end(self, batch, logs=None):
        if self.cutoff > self.epochs_completed or self.cutoff == -1:
            weights = self.model.get_weights()[0].copy()
            biases = self.model.get_weights()[1].copy()

            if self.do_weights:
                weights = self.__stroke(weights)

            if self.do_biases:
                biases = self.__stroke(biases)

            self.model.set_weights([weights, biases])

        if self.decay is not None and .01 < self.v_ratio * self.decay < 1.0:
            self.v_ratio = self.v_ratio * self.decay

        self.epochs_completed += 1
