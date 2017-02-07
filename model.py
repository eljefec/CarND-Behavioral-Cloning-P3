import load as ld
import pre
import build as bld

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

# Load data.
(X_train, y_train) = ld.load_data('train.p', 'd:\\carcapture')

# Normalize features using Min-Max scaling between -0.5 and 0.5.
X_train = pre.normalize(X_train)

# Split test data.
(X_train, X_test, y_train, y_test) = pre.split(X_train, y_train, 0.2)

# Shuffle training data.
(X_train, y_train) = pre.shuffle(X_train, y_train)

# Build Keras model.
image_shape = X_train.shape[1:]
model = bld.build_model(image_shape)

print('Built model')

# Train model.
