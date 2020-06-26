import numpy as np
import keras.backend as K

import keras
import keras.layers as layers
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.initializers import glorot_uniform

from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

"""
network
"""

def actor_mlp():

    inp = layers.Input(shape=(900,), name="input_x")
    # adv = layers.Input(shape=(1,), name="advantages")
    dense_1 = layers.Dense(32, 
                     activation="relu", 
                     use_bias=True,
                     kernel_initializer=glorot_uniform(seed=42),
                     name="dense_1")(inp)
    dense_2 = layers.Dense(32, 
                     activation="relu", 
                     use_bias=True,
                     kernel_initializer=glorot_uniform(seed=42),
                     name="dense_2")(dense_1)
    out = layers.Dense(4, 
                       activation="softmax", 
                       kernel_initializer=glorot_uniform(seed=42),
                       use_bias=True,
                       name="out")(dense_2)
    mlp_model = Model(inputs=inp, outputs=out)

    return mlp_model

def actor_cnn():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(30,30,1)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    return model

def critic_mlp():

    mlp_model = Sequential()

    mlp_model.add(Dense(512, activation='tanh', input_dim=900, init='uniform'))
    mlp_model.add(Dense(64, activation='tanh', init='uniform'))
    mlp_model.add(Dense(1, activation='linear', init='uniform'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    mlp_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    return mlp_model

