from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

def build_model(input_shape):
    model = Sequential()

    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode = 'valid',
                            input_shape = input_shape))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(0.5))

    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode = 'valid'))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(0.5))

    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode = 'valid'))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(0.5))

    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(256))

    model.add(Activation('relu'))

    model.add(Dense(256))

    model.add(Activation('relu'))

    model.add(Dense(1))

    return model
