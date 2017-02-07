# Load data.
import load as ld

(X_train, y_train) = ld.load_data('train.p', 'd:\\carcapture')

print('shape:', X_train.shape, y_train.shape)

# Normalize features using Min-Max scaling between -0.5 and 0.5.

# Split test data.

# Shuffle training data.

import pickle
import numpy as np
import math

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

print('Modules loaded.')

# Build Keras model.

# Traing model.
