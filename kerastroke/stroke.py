import random
import numpy as np

from keras.callbacks import Callback


class Stroke(Callback):
    """
    Re-initialize a percentage of weights in a layer or model at the end of every epoch to improve generalization.
    
    Arguments:
        value: If set, re-initialized weights will be set to this value, rather than a random one
        low_bound: When weights are re-initialized, this will be their lowest possible value
        high_bound: When weights are re-initialized, this will be their highest possible value
        volatility_ratio: The percentage of weights that will be re-initialized after every epoch
        cutoff: The number of epochs that Stroke will be performed
        decay: Every epoch, v_ratio is multiplied by this number. decay can be greater than 1.0, but v_ratio will
               never exceed 1.0
        indeces: A list of integers specifying the indeces of layers that Stroke will be performed on, rather than
                 the model as a whole
        biases: If true, Stroke will also be performed on biases
    """

    def __init__(self,
                 value: float = None,
                 low_bound: float = -.05,
                 high_bound: float = .05,
                 volatility_ratio: float = .05,
                 cutoff: int = -1,
                 decay: float = None,
                 indeces=None,
                 biases: bool = False):

        super(Stroke, self).__init__()

        self.value = value
        self.low_bound = low_bound
        self.high_bound = high_bound
        assert volatility_ratio < 1.0, "volatility_ratio must be less than 1.0"
        self.v_ratio = volatility_ratio
        self.cutoff = cutoff
        self.decay = decay
        self.indeces = indeces
        self.biases = biases

        self.epochs_completed = 0

    def on_epoch_end(self, batch, logs=None):
        if self.cutoff > self.epochs_completed or self.cutoff == -1:
            if self.indeces is None:

                weights = self.model.get_weights()[0]
                biases = self.model.get_weights()[1]

                num_weights = len(np.nditer(weights))
                num_biases = len(np.nditer(biases))

                for stricken in range(int(num_weights * self.v_ratio)):
                    if self.value is None:
                        weights[tuple(map(lambda x: np.random.randint(0, x), weights.shape))] = random.uniform(
                                self.low_bound, self.high_bound)
                    else:
                        weights[tuple(map(lambda x: np.random.randint(0, x), weights.shape))] = self.value

                if self.biases:
                    for stricken in range(int(num_biases * self.v_ratio)):
                        if self.value is None:
                            biases[tuple(map(lambda x: np.random.randint(0, x), weights.shape))] = random.uniform(
                                    self.low_bound, self.high_bound)
                        else:
                            biases[tuple(map(lambda x: np.random.randint(0, x), weights.shape))] = self.value

                self.model.set_weights([weights, biases])

            else:
                for index in self.indeces:
                    weights = self.model.get_layer(index=index).get_weights()[0]
                    biases = self.model.get_layer(index=index).get_weights()[1]

                    num_weights = len(np.diter(weights))
                    num_biases = len(np.nditer(biases))

                    for stricken in range(int(num_weights * self.v_ratio)):
                        if self.value is None:
                            weights[tuple(map(lambda x: np.random.randint(0, x), weights.shape))] = random.uniform(
                                self.low_bound, self.high_bound)
                        else:
                            weights[tuple(map(lambda x: np.random.randint(0, x), weights.shape))] = self.value

                    if self.biases:
                        for stricken in range(int(num_biases * self.v_ratio)):
                            if self.value is None:
                                biases[tuple(map(lambda x: np.random.randint(0, x), weights.shape))] = random.uniform(
                                    self.low_bound, self.high_bound)
                            else:
                                biases[tuple(map(lambda x: np.random.randint(0, x), weights.shape))] = self.value

                    self.model.get_layer(index=index).set_weights([weights, biases])

        if self.decay is not None and .01 < self.v_ratio < 1.0:
            self.v_ratio = self.v_ratio * self.decay
