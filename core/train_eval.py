import torch
from core.knn import get_knn

def train(model, X_train, y_train, optimizer, loss_fn, K):
    """
    Trains the model on the given data.

    Args:
        model: The model to be trained.
        X_train: The training data.
        y_train: The labels for the training data.
        optimizer: The optimizer to use for training.
        loss_fn: The loss function to use for training.
        K: The number of nearest neighbors to consider when training the model.

    Returns:
        The average loss over the training data.
    """
    model.train()
    total_loss = 0.0

    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]

        neighbors, labels = get_knn(x, X_train, y_train, K)

        pred = model(x, neighbors, labels).unsqueeze(0)
        loss = loss_fn(pred, y.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(X_train)


def evaluate(model, X_test, y_test, X_train, y_train, K):
    """
    Evaluates the model on the given data.

    Args:
        model: The model to be evaluated.
        X_test: The test data.
        y_test: The labels for the test data.
        X_train: The training data.
        y_train: The labels for the training data.
        K: The number of nearest neighbors to consider when evaluating the model.

    Returns:
        The accuracy of the model on the test data.
    """
    model.eval()
    correct = 0

    with torch.no_grad():
        for i in range(len(X_test)):
            x = X_test[i]
            y = y_test[i]

            neighbors, labels = get_knn(x, X_train, y_train, K)
            pred = model(x, neighbors, labels)

            if pred.argmax() == y:
                correct += 1

    return correct / len(X_test)