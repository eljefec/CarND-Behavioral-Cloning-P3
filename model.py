import argparse
import history as hist
import load as ld
import pre
import build as bld
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Train a neural network model.
#
# Parameters:
#     model_name: 'simple' or 'nvda'
#     model_id: Identifier for generated model files.
#     capture_root: Root folder containing training examples. 
#     steering_correction: Steering correction value for left and right images.
#     dropout: Dropout probability.
#     center_only: Use the center image only.
#     nb_epoch: Number of epochs.
#
# Returns Keras history object.
#
def train_model(model_name, model_id, capture_root, steering_correction, dropout, center_only, nb_epoch):
    print()
    print('Train model. id=[{}], capture_root=[{}], steering_correction=[{}], dropout=[{}], center_only=[{}], nb_epoch=[{}].'.format(model_id, capture_root, steering_correction, dropout, center_only, nb_epoch))

    # Load data.
    (X_train, y_train) = ld.load_data(model_id, capture_root, steering_correction, center_only)

    # Split test data.
    (X_train, X_test, y_train, y_test) = pre.split(X_train, y_train, 0.2)

    # Shuffle training data.
    (X_train, y_train) = pre.shuffle(X_train, y_train)

    # Build Keras model.
    image_shape = X_train.shape[1:]
    model_builder = bld.ModelBuilder(dropout, image_shape)
    model = model_builder.build_model(model_name)

    print('Built model. [{}]'.format(model_name))

    # Fix error with TF and Keras
    import tensorflow as tf
    tf.python.control_flow_ops = tf

    # Compile and train the model here.
    model.compile(optimizer = Adam(lr = 0.0001), loss = 'mse', metrics = ['accuracy'])

    checkpoint_path = get_model_filename(model_name, model_id, steering_correction, dropout, center_only, suffix = 'e{epoch:02d}-vl{val_loss:.2f}')

    # Set up callbacks. Stop early if the model does not improve. Save model checkpoints.
    # Source: http://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=False, verbose=0),
    ]

    history = model.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epoch, validation_split=0.2, callbacks=callbacks)

    model_filename = get_model_filename(model_name, model_id, steering_correction, dropout, center_only)
    model.save(model_filename)

    print('Saved model. [{}]'.format(model_filename))

    hist.dump_history(history, model_filename)

    return history

def make_dir_if_not_exist(path):
    import os
    if not os.path.isdir(path):
        os.mkdir(path)

# Returns a model filename including parameters.
def get_model_filename(model_name, model_id, steering_correction, dropout, center_only, suffix = '', ext = 'h5'):
    model_dir = '.\\{}'.format(model_id)
    make_dir_if_not_exist(model_dir)
    if suffix:
        suffix = '-' + suffix
    if center_only:
        center_id = '-centeronly'
    else:
        center_id = ''
    return '{}\\model-{}-{}-corr{}-drop{}{}{}.{}'.format(model_dir, model_name, model_id, steering_correction, dropout, center_id, suffix, ext)

parser = argparse.ArgumentParser(description='Build Model')
parser.add_argument('id', type=str, help='Required model identifier.')
args = parser.parse_args()

# Train a simple model with a single epoch as a smoke test.
train_model('simple', args.id, 'e:\\capture-data-archive', 0.01, 0.8, center_only = False, nb_epoch = 1)

# Train model with final parameters.
for steering_correction in [0.04]:
    for dropout in [0.2]:
        for center_only in [False]:
            train_model('nvda', args.id, 'e:\\capture-data', steering_correction, dropout, center_only, nb_epoch = 99)
