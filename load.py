# Load data.

import cv2
import pre
import os
import numpy as np
import pandas as pd
import pickle
from os.path import isfile

# Source: http://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_immediate_subdirectories(a_dir):
    return [(os.path.join(a_dir, name)) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def load_images(df, capdir, relative):
    X = []
    y = []
    for row in df.itertuples(True):
        images = []
        angles = []

        # center, left, right images
        for i in range(1, 4):
            imgpath = row[i].strip()
            if imgpath:
                if relative:
                    imgpath = os.path.join(capdir, imgpath)
                if isfile(imgpath):
                    img = cv2.imread(imgpath)
                    images.append(img)

        steering_angle = row[4]

        correction = 0.15
        if len(images) >= 1:
            angles.append(steering_angle)
        if len(images) == 2:
            print(row)
            raise ValueError('Unexpected number of images found.')
        if len(images) == 3:
            angles.append(steering_angle + correction)
            angles.append(steering_angle - correction)

        for image, angle in zip(images, angles):
            X.append(image)
            y.append(angle)

            flipped_image = cv2.flip(image, 1)
            flipped_angle = angle * -1.0
            X.append(flipped_image)
            y.append(flipped_angle)

    X = np.array(X)
    y = np.array(y)
    return (X, y)

def load_captures(capdirs, relative):
    X_list = []
    y_list = []
    csv_list = []

    for capdir in capdirs:
        csvpath = os.path.join(capdir, 'driving_log.csv')
        print(csvpath)

        # Source: https://carnd-forums.udacity.com/questions/36054925/answers/36057843
        df = pd.read_csv(csvpath)
        (X, y) = load_images(df, capdir, relative)

        print('shape:', X.shape, y.shape)

        X_list.append(X)
        y_list.append(y)
        csv_list.append(csvpath)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    print('Concatenated shape:', X.shape, y.shape)

    return (X, y, csv_list)

def load_data(filename, capture_root, relative):
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

        (X, y, csv_list) = load_captures(capdirs, relative)

        data = dict()
        data['features'] = X
        data['labels'] = y
        data['csvlist'] = csv_list

        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print('Dumped pickle. [{}]'.format(filename))
        print(csv_list)

    print('X.shape={}'.format(X.shape))

    return (X, y)
