import argparse
import load as ld
import pre
import build as bld
from keras.optimizers import Adam

parser = argparse.ArgumentParser(description='Build Model')
parser.add_argument('id', type=str, help='Optional id.')
args = parser.parse_args()

# Load data.
(X_train, y_train) = ld.load_data('udacity-flip-corr0.15-train.p', 'e:\\udacity-data', True)

# Split test data.
(X_train, X_test, y_train, y_test) = pre.split(X_train, y_train, 0.2)

# Shuffle training data.
(X_train, y_train) = pre.shuffle(X_train, y_train)

model_name = 'nvda'

# Build Keras model.
image_shape = X_train.shape[1:]
model = bld.build_model(model_name, image_shape)

print('Built model. [{}]'.format(model_name))

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

# Compile and train the model here.
model.compile(optimizer = Adam(lr = 0.0001), loss = 'mse', metrics = ['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, nb_epoch=10, validation_split=0.2)

model_filename = 'model-{}-{}.h5'.format(model_name, args.id)

model.save(model_filename)

print('Saved model. [{}]'.format(model_filename))
