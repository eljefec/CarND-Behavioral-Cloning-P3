# Load data.

import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import pickle
from os.path import isfile

# Source: http://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_immediate_subdirectories(a_dir):
    return [(os.path.join(a_dir, name)) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# https://carnd-forums.udacity.com/questions/36054925/answers/36057843

def load_images(df):
    X = []
    y = []
    for row in df.itertuples(True):
        imgpath = row[1]
        steering_angle = row[4]
        img = mpimg.imread(imgpath)
        X.append(img)
        y.append(steering_angle)
    X = np.array(X)
    y = np.array(y)
    return (X, y)

def load_captures(capdirs):
    X_list = []
    y_list = []
    csv_list = []

    for capdir in capdirs:
        csvpath = os.path.join(capdir, 'driving_log.csv')
        print(csvpath)

        df = pd.read_csv(csvpath)
        (X, y) = load_images(df)

        print('shape:', X.shape, y.shape)

        X_list.append(X)
        y_list.append(y)
        csv_list.append(csvpath)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    print('Concatenated shape:', X.shape, y.shape)

    return (X, y, csv_list)

def load_data(filename, capture_root):
    if isfile(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        X = data['features']
        y = data['labels']
        csv_list = data['csvlist']

        print('Loaded pickle. [{}]'.format(filename))
        print(csv_list)
    else:
        print('Pickle not found. Loading from captures.')

        capdirs = get_immediate_subdirectories(capture_root)

        (X, y, csv_list) = load_captures(capdirs)

        data = dict()
        data['features'] = X
        data['labels'] = y
        data['csvlist'] = csv_list

        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print('Dumped pickle. [{}]'.format(filename))
        print(csv_list)

    return (X, y)
