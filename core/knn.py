import torch

def get_knn(x, X_train, y_train, K):
    """
    This function takes in an input x, training data X_train and y_train, 
    and the number of nearest neighbors K to find. It returns the K nearest 
    neighbors of x in X_train and their corresponding labels in y_train.
    """
    distances = torch.norm(X_train - x, dim=1)
    idx = torch.topk(distances, K, largest=False).indices
    return X_train[idx], y_train[idx]