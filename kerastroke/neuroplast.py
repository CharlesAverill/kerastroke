import numpy as np

import random

from tensorflow.keras.callbacks import Callback


class NeuroPlast(Callback):
    """
    Re-initialize weights that are 0 or close to 0, so that models can operate at 'full capacity'
    """

    def __init__(self,
                 set_value: float = None,
                 min_value: float = -.01,
                 max_value: float = .01,
                 low_bound: float = -.05,
                 high_bound: float = .05,
                 cutoff: int = -1,
                 do_weights: bool = True,
                 do_biases: bool = False):
        """
        :param set_value: re-initialized weights will be set to this value, rather than a random one
        :param min_value: lowest value a weight/bias can be to be operated on
        :param max_value: highest value a weight/bias can be to be operated on
        :param low_bound: low bound for weight re-initialization
        :param high_bound: high bound for weight re-initialization
        :param cutoff: number of epochs to perform PBWOs
        :param do_weights: perform neuroplast on weights
        :param do_biases: perform neuroplast on biases
        """
        super(NeuroPlast, self).__init__()
        
        self.value = set_value
        self.min_value = min_value
        self.max_value = max_value
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.cutoff = cutoff
        self.do_weights = do_weights
        self.do_biases = do_biases

        self.epochs_completed = 0

    def __neuroplast(self, matrix):
        """
        :param matrix: np_array to perform neuroplast on
        :return: modified matrix
        """
        
        matrix_shape = matrix.shape
        flat_matrix = matrix.flatten()
        ls_matrix = flat_matrix.tolist()

        for i in range(len(ls_matrix)):
            weight = ls_matrix[i]
            if self.min_value <= weight <= self.max_value:
                if self.value:
                    ls_matrix[i] = self.value
                else:
                    ls_matrix[i] = random.uniform(self.low_bound, self.high_bound)

        matrix = np.array(ls_matrix)
        matrix = np.reshape(matrix, matrix_shape)
        
        return matrix
    
    def on_epoch_end(self, batch, logs=None):
        if self.cutoff > self.epochs_completed or self.cutoff == -1:
            weights = self.model.get_weights()[0].copy()
            biases = self.model.get_weights()[1].copy()

            if self.do_weights:
                weights = self.__neuroplast(weights)

            if self.do_biases:
                biases = self.__neuroplast(biases)

            self.model.set_weights([weights, biases])

        self.epochs_completed += 1
