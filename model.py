# Load data.

# Source: http://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
import os
def get_immediate_subdirectories(a_dir):
    return [(os.path.join(a_dir, name)) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

capdirs = get_immediate_subdirectories('d:\\carcapture')

# https://carnd-forums.udacity.com/questions/36054925/answers/36057843
import numpy as np
import pandas as pd

import matplotlib.image as mpimg

def load_images(df):
    X = []
    y = []
    for row in df.itertuples(True):
        # print(row[1])
        # print(row[4])
        imgpath = row[1]
        steering_angle = row[4]
        img = mpimg.imread(imgpath)
        X.append(img)
        y.append(steering_angle)
    X = np.array(X)
    y = np.array(y)
    return (X, y)

X_list = []
y_list = []

for capdir in capdirs:
    csvpath = os.path.join(capdir, 'driving_log.csv')
    print(csvpath)
    # contents = np.genfromtxt(csvpath, delimiter=',', skip_header=1, usecols=(1,), unpack=True, dtype=None)
    df = pd.read_csv(csvpath)
    # print(df)
    (X, y) = load_images(df)
    print('shape:', X.shape, y.shape)
    X_list.append(X)
    y_list.append(y)

X_train = np.concatenate(X_list)
y_train = np.concatenate(y_list)

print('shape:', X_train.shape, y_train.shape)

from os.path import isfile
import pickle

filename = 'train.p'
data = dict()
data['features'] = X_train
data['labels'] = y_train

if not isfile(filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

with open(filename, 'rb') as f:
    data = pickle.load(f)

X_train = data['features']
y_train = data['labels']

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
