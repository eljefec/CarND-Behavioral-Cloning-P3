import argparse
import history as hist
import load as ld
import pre
import build as bld
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model_name, model_id, capture_root, steering_correction, dropout, nb_epoch):
    print('Train model. id=[{}], capture_root=[{}], steering_correction=[{}], dropout=[{}]'.format(model_id, capture_root, steering_correction, dropout))

    # Load data.
    (X_train, y_train) = ld.load_data(model_id, capture_root, steering_correction, center_only = False)

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

    checkpoint_path = get_model_filename(model_name, model_id, steering_correction, dropout, suffix = 'checkpoint')

    # Source: http://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=0),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epoch, validation_split=0.2, callbacks=callbacks)

    model_filename = get_model_filename(model_name, model_id, steering_correction, dropout)
    model.save(model_filename)

    print('Saved model. [{}]'.format(model_filename))

    hist.dump_history(history, model_filename)

    return history

def get_model_filename(model_name, model_id, steering_correction, dropout, suffix = '', ext = 'h5'):
    if suffix:
        suffix = '-' + suffix
    return 'model-{}-{}-corr{}-drop{}{}.{}'.format(model_name, model_id, steering_correction, dropout, suffix, ext)

parser = argparse.ArgumentParser(description='Build Model')
parser.add_argument('id', type=str, help='Optional id.')
args = parser.parse_args()

train_model('simple', args.id, 'e:\\capture-data-archive', 0.01, 0.8, nb_epoch = 1)

for steering_correction in [0.04, 0.08, 0.12]:
    for dropout in [0.2]:
        train_model('nvda', args.id, 'e:\\capture-data', steering_correction, dropout, nb_epoch = 30)
