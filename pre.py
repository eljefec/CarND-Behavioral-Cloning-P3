from sklearn.model_selection import train_test_split
import sklearn.utils

def normalize(image_data):
    a = -0.5
    b = 0.5
    rgb_min = 0
    rgb_max = 255
    return a + (((image_data - rgb_min) * (b - a)) / (rgb_max - rgb_min))

def split(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return (X_train, X_test, y_train, y_test)

def shuffle(X, y):
    return sklearn.utils.shuffle(X, y)
