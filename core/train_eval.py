import torch
from core.knn import get_knn

def train(model, X_train, y_train, optimizer, loss_fn, K):
    model.train()
    total_loss = 0.0

    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]

        # Get neighbors
        neighbors, labels = get_knn(x, X_train, y_train, K)

        # Add batch dimension
        x = x.unsqueeze(0)                  # (1, dim)
        neighbors = neighbors.unsqueeze(0)  # (1, K, dim)
        labels = labels.unsqueeze(0)        # (1, K)

        # Forward
        pred, _ = model(x, neighbors, labels)  # (1, num_classes)

        # Compute loss
        loss = loss_fn(pred, y.unsqueeze(0))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(X_train)


def evaluate(model, X_test, y_test, X_train, y_train, K):
    model.eval()
    correct = 0

    with torch.no_grad():
        for i in range(len(X_test)):
            x = X_test[i]
            y = y_test[i]

            # Get neighbors from TRAIN set
            neighbors, labels = get_knn(x, X_train, y_train, K)

            # Add batch dimension
            x = x.unsqueeze(0)
            neighbors = neighbors.unsqueeze(0)
            labels = labels.unsqueeze(0)

            # Forward
            pred, _ = model(x, neighbors, labels)

            if pred.argmax(dim=1).item() == y.item():
                correct += 1

    return correct / len(X_test)