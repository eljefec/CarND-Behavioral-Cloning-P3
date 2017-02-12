import argparse
import load as ld
import matplotlib.pyplot as plt
from keras.models import load_model

def plot_histogram(y, bins, titleprefix):
    plt.hist(y, bins)
    plt.xlabel(titleprefix + ' Label')
    plt.ylabel('Count')
    plt.title('Histogram of ' + titleprefix + ' Labels')
    
parser = argparse.ArgumentParser(description='Predictions')
parser.add_argument('model', type=str, help='Path to model definition h5. Model should be on the same path.')
args = parser.parse_args()

model = load_model(args.model)

# Load data.
(X_train, y_train) = ld.load_data('udacity-flip-train.p', 'e:\\udacity-data', True)

predictions = model.predict(X_train)
print(predictions.shape)

plt.subplot(211)
plot_histogram(y_train, 40, 'Train')
plt.subplot(212)
plot_histogram(predictions, 40, 'Predictions')
plt.show()
