#!/usr/bin/env python3
"""Run MLP experiments with different hyperparameters.

Uses a synthetic 10-class classification problem (20k samples).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import goodseed


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    X, y = make_classification(
        n_samples=20000, n_features=40, n_informative=15, n_redundant=5,
        n_classes=10, n_clusters_per_class=2, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (
        torch.FloatTensor(X_train), torch.FloatTensor(X_test),
        torch.LongTensor(y_train), torch.LongTensor(y_test),
    )


def train_one(hidden_size, lr, dropout, num_epochs):
    run = goodseed.Run(name="synth-mlp", project="examples")

    run.log_configs({
        "model/hidden_size": hidden_size,
        "model/dropout": dropout,
        "training/lr": lr,
        "training/epochs": num_epochs,
        "dataset": "make_classification-20k",
    })

    X_train, X_test, y_train, y_test = load_data()
    model = MLP(40, hidden_size, 10, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_step = 0
    for epoch in range(num_epochs):
        model.train()
        idx = torch.randperm(len(X_train))
        total_loss, correct, total, n = 0.0, 0, 0, 0

        for i in range(0, len(X_train), 64):
            bi = idx[i:i+64]
            out = model(X_train[bi])
            loss = criterion(out, y_train[bi])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y_train[bi]).sum().item()
            total += len(bi)
            n += 1
            batch_step += 1
            if batch_step % 10 == 0:
                run.log_metrics({"train/batch_loss": loss.item()}, step=batch_step)

        train_loss = total_loss / n
        train_acc = correct / total
        run.log_metrics({"train/loss": train_loss, "train/accuracy": train_acc}, step=epoch + 1)

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(X_test)
                test_loss = criterion(test_out, y_test).item()
                test_acc = (test_out.argmax(1) == y_test).sum().item() / len(y_test)
            run.log_metrics({"test/loss": test_loss, "test/accuracy": test_acc}, step=epoch + 1)

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1:3d}/{num_epochs}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    run.log_configs({"summary/final_test_acc": test_acc})
    print(f"  -> final test_acc={test_acc:.4f}")
    run.close()
    return test_acc


def main():
    configs = [
        (32,  0.001, 0.1, 200),
        (64,  0.003, 0.1, 200),
        (128, 0.003, 0.2, 200),
        (128, 0.01,  0.2, 200),
        (256, 0.003, 0.3, 200),
    ]
    print(f"Running {len(configs)} experiments...\n")
    results = []
    for i, (h, lr, drop, ep) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] hidden={h}, lr={lr}, dropout={drop}")
        acc = train_one(h, lr, drop, ep)
        results.append((h, lr, drop, acc))

    print("\n--- Summary ---")
    results.sort(key=lambda x: x[3], reverse=True)
    for h, lr, drop, acc in results:
        print(f"  hidden={h}, lr={lr}, dropout={drop} -> {acc:.4f}")


if __name__ == "__main__":
    main()
