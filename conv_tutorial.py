from __future__ import absolute_import, division, print_function

import numpy as np
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

def cnn_model_fn(features, labels, mode):
    model = Sequential()
    model.add(Conv2D(input_shape))