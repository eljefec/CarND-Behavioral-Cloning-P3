from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Cropping2D

def build_model(name, input_shape):
    if name == 'simple':
        return build_model_simple(input_shape)
    elif name == 'nvda':
        return build_model_nvda(input_shape)
    elif name == 'complex':
        return build_model_complex(input_shape)
    else:
        raise ValueError('Invalid model name [{}]'.format(name))

def build_model_simple(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = input_shape))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(1))
    return model

def add_conv(model, nb_filter, nb_row, nb_col, pool):
    model.add(Convolution2D(nb_filter, nb_row, nb_col, border_mode = 'valid'))
    if (pool):
        model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Activation('relu'))
    print(model.outputs)

def build_model_nvda(input_shape):
    model = Sequential()

    # Crop out sky (50 pixels) and car hood (20 pixels).
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape = input_shape))

    # Normalize input.
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    # Let this layer modify color.
    # This layer is inspired by "Color Space Transformation Network" at https://arxiv.org/ftp/arxiv/papers/1511/1511.01064.pdf.
    model.add(Convolution2D(3, 1, 1, border_mode = 'valid'))

    model.add(AveragePooling2D())

    add_conv(model, 24, 5, 5, pool = True)
    add_conv(model, 36, 5, 5, pool = False)
    add_conv(model, 48, 5, 5, pool = True)
    add_conv(model, 64, 3, 3, pool = False)
    add_conv(model, 64, 3, 3, pool = False)

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

def build_model_complex(input_shape):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = input_shape))

    # Let this layer modify color.
    # This layer is inspired by "Color Space Transformation Network" at https://arxiv.org/ftp/arxiv/papers/1511/1511.01064.pdf.
    model.add(Convolution2D(3, 1, 1,
                            border_mode = 'valid',
                            input_shape = input_shape))

    conv_pool_count = 3
    for i in range(conv_pool_count):
        model.add(Convolution2D(32, 3, 3,
                                border_mode = 'valid'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    conv_count = 3
    for i in range(conv_count):
        model.add(Convolution2D(64 * (2 ** i), 5, 5, border_mode = 'valid'))
        model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    print(model.outputs)

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(1))

    return model
