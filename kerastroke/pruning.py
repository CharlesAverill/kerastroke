import numpy as np

from tensorflow.keras.callbacks import Callback


class Pruning(Callback):
    """
    Set weights that fall between a certain bound to a certain value. Typically used to reduce model size by setting
    weights that are close to 0.0, to 0.0.
    """

    def __init__(self,
                 set_value: float = 0.0,
                 min_value: float = -.2,
                 max_value: float = .2,
                 cutoff: int = -1,
                 do_weights: bool = True,
                 do_biases: bool = False):
        """
        :param set_value: The value that pruned weights will be set to
        :param min_value: The lowest value a weight/bias can be to be oeprated on
        :param max_value: The highest value a weight/bias can be to be operated on
        :param cutoff: number of epochs to perform PBWOs
        :param do_weights: perform pruning on weights
        :param do_biases: perform pruning on biases
        """
        super(Pruning, self).__init__()

        self.value = set_value
        self.min_value = min_value
        self.max_value = max_value
        self.cutoff = cutoff
        self.do_weights = do_weights
        self.do_biases = do_biases

        self.epochs_completed = 0

    def __prune(self, matrix):
        """
        :param matrix: np_array to perform pruning on
        :return: modified matrix
        """

        matrix_shape = matrix.shape
        flat_matrix = matrix.flatten()
        ls_matrix = flat_matrix.tolist()

        for i in range(len(ls_matrix)):
            weight = ls_matrix[i]
            if self.min_value <= weight <= self.max_value:
                ls_matrix[i] = self.value

        matrix = np.array(ls_matrix)
        matrix = np.reshape(matrix, matrix_shape)

        return matrix

    def on_epoch_end(self, batch, logs=None):
        if self.cutoff > self.epochs_completed or self.cutoff == -1:
            weights = self.model.get_weights()[0].copy()
            biases = self.model.get_weights()[1].copy()

            if self.do_weights:
                weights = self.__prune(weights)

            if self.biases:
                biases = self.__prune(biases)

            self.model.set_weights([weights, biases])

        self.epochs_completed += 1
