from sklearn.model_selection import train_test_split
import sklearn.utils

# Split training data into training and test sets.
def split(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return (X_train, X_test, y_train, y_test)

# Shuffle training data.
def shuffle(X, y):
    return sklearn.utils.shuffle(X, y)
