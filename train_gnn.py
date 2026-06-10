import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config
from data_loader import get_data_loaders
from models.gnn import TabularGNN


def train_model():

    print("⏳ Loading dataset...")

    train_loader, val_loader, test_loader, input_dim = get_data_loaders()

    model = TabularGNN(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-2
    )

    print(f"🚀 Training GNN on {config.DEVICE}")

    num_epochs = config.EPOCHS
    patience = 3

    best_val_accuracy = 0
    patience_counter = 0

    # ✅ STORE LOSSES
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        # -------- TRAIN --------
        model.train()

        total_loss = 0
        correct_train = 0
        total_train = 0

        for X, y in train_loader:
            X = X.to(config.DEVICE)
            y = y.to(config.DEVICE)

            X = X + 0.01 * torch.randn_like(X)

            optimizer.zero_grad()
            outputs = model(X)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()

        # ✅ AVERAGE TRAIN LOSS
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_acc = 100 * correct_train / total_train

        # -------- VALIDATION --------
        model.eval()

        correct_val = 0
        total_val = 0
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(config.DEVICE)
                y = y.to(config.DEVICE)

                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()

        # ✅ AVERAGE VAL LOSS
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_acc = 100 * correct_val / total_val
        
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # -------- EARLY STOPPING --------
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0

            torch.save(model.state_dict(), "GNN_Model.pth")
            print("✅ Model improved — saved.")

        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                break

    print(f"\n🏆 Best Validation Accuracy: {best_val_accuracy:.2f}%")

    # -------- LOSS CURVE --------
    plt.figure(figsize=(8, 5))

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GNN Epoch vs Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # -------- FINAL TEST --------
    print("\n📊 Evaluating on Test Set...")

    model.load_state_dict(torch.load("GNN_Model.pth"))
    model.eval()

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(config.DEVICE)
            y = y.to(config.DEVICE)

            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            total_test += y.size(0)
            correct_test += (predicted == y).sum().item()

    test_acc = 100 * correct_test / total_test

    print(f"🎯 Test Accuracy: {test_acc:.2f}%")
    print("💾 Model saved as GNN_Model.pth")


if __name__ == "__main__":
    train_model()