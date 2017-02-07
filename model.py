import load as ld

# Load data.
(X_train, y_train) = ld.load_data('train.p', 'd:\\carcapture')

print('shape:', X_train.shape, y_train.shape)

import pre

# Normalize features using Min-Max scaling between -0.5 and 0.5.
X_train = pre.normalize(X_train)

# Split test data.
(X_train, X_test, y_train, y_test) = pre.split(X_train, y_train, 0.2)

# Shuffle training data.
(X_train, y_train) = pre.shuffle(X_train, y_train)

import pickle
import numpy as np
import math

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

print('Modules loaded.')

# Build Keras model.

# Traing model.
