import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

from data.data import load_data
from models.attention_knn import AttentionKNN
from core.train_eval import train, evaluate
from config import Config

def main():
    # ---- Create output folders ----
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # ---- Load data ----
    X_train, X_test, y_train, y_test = load_data()

    # ---- Model ----
    model = AttentionKNN(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    loss_fn = nn.CrossEntropyLoss()

    acc_list = []
    loss_list = []

    # ---- Training loop ----
    for epoch in range(Config.epochs):
        loss = train(model, X_train, y_train, optimizer, loss_fn, Config.K)
        acc = evaluate(model, X_test, y_test, X_train, y_train, Config.K)

        loss_list.append(loss)
        acc_list.append(acc)

        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")

        # ---- Log to file ----
        with open("outputs/logs/train_log.txt", "a") as f:
            f.write(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}\n")

    # ---- Save model ----
    torch.save(model.state_dict(), "outputs/models/attn_knn.pt")

    # ---- Plot accuracy ----
    plt.figure()
    plt.plot(range(Config.epochs), acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Attention KNN Accuracy")
    plt.savefig("outputs/plots/accuracy.png")

    print("✅ Training complete. Results saved in outputs/")

if __name__ == "__main__":
    main()