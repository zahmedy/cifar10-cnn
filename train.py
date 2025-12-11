from __future__ import annotations

import torch
from torch import nn, optim
from typing import Tuple

from config import CHECKPOINT_DIR, DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from data import get_dataloaders
from model import Cifar10CNN


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    # One full pass over the training data: learn weights and measure accuracy.
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    # Measure accuracy and loss without updating the model.
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = loss_fn(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train():
    # Get data loaders for train/val, build the model, and pick an optimizer/loss.
    train_loader, val_loader, _ = get_dataloaders()
    model = Cifar10CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = CHECKPOINT_DIR / "cifar10_cnn.pt"

    for epoch in range(1, EPOCHS + 1):
        # Train for one epoch, then check performance on the validation set.
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        print(
            f"[{epoch:03d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            # Keep the best-performing model so inference uses a good checkpoint.
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                best_path,
            )
            print(f"Saved new best checkpoint to {best_path} (val_acc={val_acc:.4f})")


if __name__ == "__main__":
    train()
