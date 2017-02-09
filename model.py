import load as ld
import pre
import build as bld
from keras.optimizers import Adam

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

# Load data.
(X_train, y_train) = ld.load_data('udacity-allcams-train.p', 'e:\\udacity-data', True)

# Split test data.
(X_train, X_test, y_train, y_test) = pre.split(X_train, y_train, 0.2)

# Shuffle training data.
(X_train, y_train) = pre.shuffle(X_train, y_train)

model_name = 'nvda'

# Build Keras model.
image_shape = X_train.shape[1:]
model = bld.build_model(model_name, image_shape)

print('Built model. [{}]'.format(model_name))

# Compile and train the model here.
model.compile(optimizer = Adam(lr = 0.0001), loss = 'mse', metrics = ['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, nb_epoch=10, validation_split=0.2)

model.save('model-{}.h5'.format(model_name))

print('Saved model.')
