import load as ld
import pre
import build as bld

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

# Load data.
(X_train, y_train) = ld.load_data('train.p', 'd:\\carcapture')

# Normalize features using Min-Max scaling between -0.5 and 0.5.
group_size = 1024
for i in range(0, X_train.shape[0], group_size):
    X_train[i:i+group_size] = pre.normalize(X_train[i:i+group_size])

# Split test data.
(X_train, X_test, y_train, y_test) = pre.split(X_train, y_train, 0.2)

# Shuffle training data.
(X_train, y_train) = pre.shuffle(X_train, y_train)

# Build Keras model.
image_shape = X_train.shape[1:]
model = bld.build_model(image_shape)

print('Built model.')

# Compile and train the model here.
model.compile('adam', 'mse', ['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, nb_epoch=10, validation_split=0.2)

model.save('model.h5')

print('Saved model.')
