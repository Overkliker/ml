# import numpy as np
#
# import keras
# from keras.layers import Dense, Input
# import tensorflow as tf
# import keras.backend as K
# import visualkeras
#
#
# def binary_step(x):
#     return K.cast(K.greater_equal(x, 0), K.floatx())
#
#
# model = keras.Sequential()
#
#
# model.add(Dense(units=2, input_shape=(4,), activation='relu'))
# model.add(Dense(units=1, input_shape=(2,), activation=binary_step))
#
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#
# visualkeras.graph_view(model, to_file='gg.png')

def prt(a, *b):
    print(a)
    print(b)

a = [1, 3, 4]

prt(*a)