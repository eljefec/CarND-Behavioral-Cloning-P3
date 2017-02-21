from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Cropping2D

class ModelBuilder:
    def __init__(self, dropout, input_shape):
        self.dropout = dropout
        self.input_shape = input_shape

    # Build a neural network architecture.
    # Supported name values: 'simple', 'nvda'
    def build_model(self, name):
        if name == 'simple':
            return self.build_model_simple()
        elif name == 'nvda':
            return self.build_model_nvda()
        else:
            raise ValueError('Invalid model name [{}]'.format(name))

    # Build a simple neural network architecture.
    def build_model_simple(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = self.input_shape))
        model.add(AveragePooling2D(pool_size = (2, 2)))
        model.add(AveragePooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dense(1))
        return model

    # Add a convolutional layer and a relu activation layer.
    # 
    # Parameters:
    #   model: Keras model
    #   nb_filter: Filter depth
    #   nb_row: Number of rows in filter
    #   nb_col: Number of columns in filter
    #   stride: If True, use 2x2 stride for convolutional filter. If False, use 1x1 stride.
    #
    def add_conv(self, model, nb_filter, nb_row, nb_col, stride):
        if (stride):
            model.add(Convolution2D(nb_filter, nb_row, nb_col, subsample = (2, 2), border_mode = 'valid'))
        else:
            model.add(Convolution2D(nb_filter, nb_row, nb_col, border_mode = 'valid'))
        model.add(Activation('relu'))

    # Build CNN architecture based on Nvidia's model.
    # Source: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    def build_model_nvda(self):
        model = Sequential()

        # Crop out sky (50 pixels) and car hood (20 pixels).
        model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape = self.input_shape))

        # Normalize input.
        model.add(Lambda(lambda x: x / 255.0 - 0.5))

        # Let this layer modify color.
        # This layer is inspired by "Color Space Transformation Network" at https://arxiv.org/ftp/arxiv/papers/1511/1511.01064.pdf.
        model.add(Convolution2D(3, 1, 1, border_mode = 'valid'))

        # Convolutional layers.
        self.add_conv(model, 24, 5, 5, stride = True)
        self.add_conv(model, 36, 5, 5, stride = True)
        self.add_conv(model, 48, 5, 5, stride = True)
        self.add_conv(model, 64, 3, 3, stride = True)
        self.add_conv(model, 64, 3, 3, stride = True)

        model.add(Flatten())

        # Fully connected layers.
        model.add(Dropout(self.dropout))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(50, activation = 'relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))

        return model
