from __future__ import print_function
import numpy as np

# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

from services.constants import *

"""
Module for:
  - creating neural network
  - training neural network
"""


def create_ann():
    """Implementacija veštačke neuronske mreže. Aktivaciona funkcija je sigmoid.
    """
    ann = Sequential()
    ann.add(Dense(NEURONS_IN_HIDDEN_LAYER, input_dim=NEURONS_IN_INPUT_LAYER, activation='sigmoid'))
    ann.add(Dense(NEURONS_IN_OUTPUT_LAYER, activation='sigmoid'))
    return ann


def train_ann(ann, x_train, y_train):
    """Obucavanje vestacke neuronske mreze"""
    x_train = np.array(x_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=False)

    return ann
