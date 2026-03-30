import torch
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(max_samples=5000, test_size=0.2, random_state=42):
    """
    CIFAR-10 with pretrained ResNet18 features instead of raw pixels.
    """
    from torchvision import datasets, transforms, models
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    trainset = datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform)

    # Combine
    all_data = torch.utils.data.ConcatDataset([trainset, testset])

    # Subsample
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(all_data), min(max_samples, len(all_data)), replace=False)
    subset = torch.utils.data.Subset(all_data, idx)

    # Load pretrained ResNet18 as feature extractor
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()  # Remove classification head → outputs 512-dim
    resnet.eval()

    loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=False)

    features = []
    labels   = []

    print("[CIFAR-10] Extracting ResNet18 features...")
    with torch.no_grad():
        for imgs, lbls in loader:
            feats = resnet(imgs)  # (batch, 512)
            features.append(feats.numpy())
            labels.append(lbls.numpy())

    X = np.concatenate(features, axis=0).astype(np.float32)  # (N, 512)
    y = np.concatenate(labels, axis=0).astype(np.int64)

    # Normalize
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X    = (X - mean) / std

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
