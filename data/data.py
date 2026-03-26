import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data():
    """
    Loads the MNIST dataset and splits it into a training set and a test set

    Returns:
        tuple: (X_train, X_test, y_train, y_test) where X_train and X_test are the
        training and test data and y_train and y_test are the corresponding labels
    """
    data = load_digits()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)

    return train_test_split(X, y, test_size=0.2, random_state=42)