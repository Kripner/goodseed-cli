#!/usr/bin/env python3
"""Run multiple MLP experiments with different hyperparameters.

This script runs several experiments to generate test data for the frontend.
"""

import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import goodseed


class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def load_data():
    """Load and preprocess the Iris dataset."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test


def train_one(hidden_size: int, learning_rate: float, batch_size: int):
    """Train one experiment with given hyperparameters."""
    input_size = 4
    num_classes = 3
    num_epochs = 100

    run = goodseed.Run(
        experiment_name="iris-mlp",
        project="examples",
    )

    # Log configuration
    run.log_configs({
        "model/type": "MLP",
        "model/input_size": input_size,
        "model/hidden_size": hidden_size,
        "model/num_classes": num_classes,
        "training/learning_rate": learning_rate,
        "training/num_epochs": num_epochs,
        "training/batch_size": batch_size,
        "training/optimizer": "Adam",
        "dataset/name": "iris",
    })

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Create model
    model = SimpleMLP(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        indices = torch.randperm(len(X_train))
        total_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i : i + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        avg_loss = total_loss / (len(X_train) / batch_size)
        train_accuracy = correct / total

        run.log_metrics(
            {
                "train/loss": avg_loss,
                "train/accuracy": train_accuracy,
            },
            step=epoch + 1,
        )

        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test).item()
                _, predicted = torch.max(test_outputs.data, 1)
                test_accuracy = (predicted == y_test).sum().item() / len(y_test)

            run.log_metrics(
                {
                    "test/loss": test_loss,
                    "test/accuracy": test_accuracy,
                },
                step=epoch + 1,
            )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        final_accuracy = (predicted == y_test).sum().item() / len(y_test)

    run.log_configs({
        "summary/final_accuracy": final_accuracy,
    })

    run.close()

    print(f"  Run {run.run_name}: hidden={hidden_size}, lr={learning_rate}, batch={batch_size} -> acc={final_accuracy:.4f}")
    return final_accuracy


def main():
    """Run experiments with different hyperparameter combinations."""
    # Define hyperparameter grid
    hidden_sizes = [8, 16, 32]
    learning_rates = [0.001, 0.01, 0.05]
    batch_sizes = [8, 16]

    combinations = list(itertools.product(hidden_sizes, learning_rates, batch_sizes))
    print(f"Running {len(combinations)} experiments...\n")

    results = []
    for i, (hidden, lr, batch) in enumerate(combinations, 1):
        print(f"[{i}/{len(combinations)}]")
        acc = train_one(hidden, lr, batch)
        results.append((hidden, lr, batch, acc))

    print("\n--- Summary ---")
    results.sort(key=lambda x: x[3], reverse=True)
    for hidden, lr, batch, acc in results[:5]:
        print(f"  hidden={hidden}, lr={lr}, batch={batch} -> {acc:.4f}")


if __name__ == "__main__":
    main()

