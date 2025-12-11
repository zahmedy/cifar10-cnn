import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from config import CLASSES, DEVICE, MEAN, STD, CHECKPOINT_DIR
from data import get_dataloaders
from model import Cifar10CNN


def _load_checkpoint(path: Path) -> Cifar10CNN:
    # Restore the trained model weights from disk.
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}. Train the model first.")

    checkpoint = torch.load(path, map_location=DEVICE)
    model = Cifar10CNN()
    try:
        # If the model architecture changed after training, this will raise.
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is incompatible with the current model architecture. "
            "If you changed layer sizes (e.g., number of filters), retrain so a new checkpoint is saved."
        ) from exc
    model.to(DEVICE)
    model.eval()
    return model


def _preprocess_image(image_path: Path) -> torch.Tensor:
    # Resize and normalize an image so it matches the model's expected input.
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dim


@torch.no_grad()
def predict_image(model: Cifar10CNN, image_path: Path) -> Tuple[str, torch.Tensor]:
    # Run inference on a single image and return the top label and probabilities.
    tensor = _preprocess_image(image_path).to(DEVICE)
    logits = model(tensor)
    pred_idx = logits.argmax(dim=1).item()
    return CLASSES[pred_idx], logits.squeeze().softmax(dim=0)


@torch.no_grad()
def evaluate_split(model: Cifar10CNN, split: str = "test") -> Tuple[float, float]:
    # Compute simple accuracy on the requested dataset split.
    _, val_loader, test_loader = get_dataloaders()
    dataloader = test_loader if split == "test" else val_loader
    total = 0
    correct = 0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    return acc, total


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 inference helper.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "cifar10_cnn.pt",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to a single image to classify.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["test", "val"],
        help="Evaluate accuracy on a dataset split instead of a single image.",
    )
    args = parser.parse_args()

    model = _load_checkpoint(args.checkpoint)

    if args.image:
        label, probs = predict_image(model, args.image)
        print(f"Predicted: {label}")
        print(f"Probabilities: {probs}")
    elif args.eval_split:
        acc, total = evaluate_split(model, args.eval_split)
        print(f"Accuracy on {args.eval_split} split over {total} samples: {acc:.4f}")
    else:
        parser.error("Provide either --image to classify or --eval-split to run evaluation.")


if __name__ == "__main__":
    main()
