""" LSTM Model"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

np.random.seed(1)
tf.random.set_seed(2)

def lstm(X_shape, lr, n_1, dr_1):
    # model
    model = Sequential()
    model.add(LSTM(int(n_1), input_shape=(X_shape)))
    model.add(Dropout(dr_1))

    model.add(Dense(2, activation="softmax"))

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
