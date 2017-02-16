import argparse
import load as ld
import pre
import build as bld
from keras.optimizers import Adam

def train_model(model_id, capture_root, steering_correction, dropout, dropouts):
    print('Train model. id=[{}], capture_root=[{}], steering_correction=[{}], dropout=[{}]'.format(model_id, capture_root, steering_correction, dropout))

    # Load data.
    (X_train, y_train) = ld.load_data(model_id, capture_root, steering_correction)

    # Split test data.
    (X_train, X_test, y_train, y_test) = pre.split(X_train, y_train, 0.2)

    # Shuffle training data.
    (X_train, y_train) = pre.shuffle(X_train, y_train)

    model_name = 'nvda'

    # Build Keras model.
    image_shape = X_train.shape[1:]
    model_builder = bld.ModelBuilder(dropout, image_shape)
    model = model_builder.build_model(model_name, dropouts)

    print('Built model. [{}]'.format(model_name))

    model.summary()

    # Fix error with TF and Keras
    import tensorflow as tf
    tf.python.control_flow_ops = tf

    # Compile and train the model here.
    model.compile(optimizer = Adam(lr = 0.0001), loss = 'mse', metrics = ['accuracy'])

    history = model.fit(X_train, y_train, batch_size=128, nb_epoch=10, validation_split=0.2)

    model_filename = 'model-{}-{}-corr{}-drop{}-d{:d}{:d}{:d}{:d}.h5'.format(model_name, model_id, steering_correction, dropout, dropouts[0], dropouts[1], dropouts[2], dropouts[3])

    model.save(model_filename)

    print('Saved model. [{}]'.format(model_filename))

parser = argparse.ArgumentParser(description='Build Model')
parser.add_argument('id', type=str, help='Optional id.')
args = parser.parse_args()

for steering_correction in [0.14]:
    for dropout in [0.2, 0.4]:
        for dropouts in [[True, False, False, False], 
                         [False, True, False, False], 
                         [False, False, True, False], 
                         [True, True, True, False]]:
            train_model(args.id, 'e:\\capture-data', steering_correction, dropout, dropouts)
