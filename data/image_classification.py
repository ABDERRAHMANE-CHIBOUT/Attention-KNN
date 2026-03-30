import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def load_data(max_samples=5000, n_components=128, test_size=0.2, random_state=42):
    from torchvision import datasets

    trainset = datasets.CIFAR10(root='./data_cifar10', train=True, download=True)
    testset  = datasets.CIFAR10(root='./data_cifar10', train=False, download=True)

    all_images = np.concatenate([trainset.data, testset.data], axis=0)
    all_labels = np.concatenate([trainset.targets, testset.targets], axis=0)

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(all_images), min(max_samples, len(all_images)), replace=False)
    all_images = all_images[idx]
    all_labels = all_labels[idx]

    X = all_images.reshape(len(all_images), -1).astype(np.float32) / 255.0

    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X    = (X - mean) / std

    pca = PCA(n_components=n_components, random_state=random_state)
    X   = pca.fit_transform(X)

    y = all_labels.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    num_classes = 10
    print(f"[CIFAR-10] Train: {X_train.shape}, Test: {X_test.shape}, Classes: {num_classes}")

    return X_train, X_test, y_train, y_test, num_classes
