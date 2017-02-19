import matplotlib.pyplot as plt
import pickle

def plot_history(history, model_filename):
    plot.ion()

    print(history.keys())

    # Plot the training and validation loss for each epoch
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model mean squared error loss (' + model_filename + ')')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.draw()
    plt.pause(0.001)

def get_history_filename(model_filename):
    return model_filename + '-history.p'

def dump_history(history, model_filename):
    filename = get_history_filename(model_filename)

    with open(filename, 'wb') as f:
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)

    print('Dumped history. [{}]'.format(filename))


