import numpy as np
from deap import creator
from deap import tools

import keras
from keras.layers import Dense
import keras.backend as K


def binary_step(x):
    return K.cast(K.greater_equal(x, 0), K.floatx())


class PredictModel:
    def get_total_weights(self, ):
        print(self.model.get_weights())
        return len('sd')

    def __init__(self):
        self.model = keras.Sequential()

        self.model.add(Dense(units=2, input_shape=(4,), activation='relu', use_bias=True))
        self.model.add(Dense(units=1, input_shape=(2,), activation=binary_step, use_bias=True))

        self.model.compile(loss='mean_squared_error')

        for i in range(len(self.model.layers)):
            n = len(self.model.layers[i].get_weights()[0]) + 1 #+1 bias
            m = len(self.model.layers[i].get_weights()[0][0])
            weights = self.get_weights(n, m)
            self.set_weights(i, weights)

    @staticmethod
    def get_weights(n, m):
        return np.random.triangular(-1, 0, 1, size=(n, m))

    def set_weights(self, layer, weights):
        self.model.layers[layer].set_weights([weights[:-1], weights[-1]])

    def predict(self, inputs):
        return self.model.predict(inputs)




