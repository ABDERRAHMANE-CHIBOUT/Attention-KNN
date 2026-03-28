import torch

def get_knn(x, X_train, y_train, K):
    """
    Returns K nearest neighbors using cosine similarity.
    Works well for NLP embeddings (TF-IDF, SBERT).
    """

    # Normalize vectors
    x_norm = x / (x.norm() + 1e-8)
    X_norm = X_train / (X_train.norm(dim=1, keepdim=True) + 1e-8)

    # Cosine similarity
    similarity = torch.matmul(X_norm, x_norm)

    # Exclude self-match (important during training)
    similarity[similarity == 1.0] = -float('inf')

    # Get top K most similar
    idx = torch.topk(similarity, K, largest=True).indices

    return X_train[idx], y_train[idx]