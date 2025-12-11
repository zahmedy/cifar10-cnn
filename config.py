from pathlib import Path
import os
import torch


def _select_device() -> torch.device:
    """Prefer CUDA, then MPS, and finally CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Paths
# BASE_DIR points at this folder; we keep data and saved models alongside it.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
# Change these knobs to trade training speed vs. model quality.
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.1
NUM_WORKERS = min(4, os.cpu_count() or 1)
NUM_CLASSES = 10
SEED = 42

# Normalization stats for CIFAR-10
# These numbers center/scale images so training is stable.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Human-readable class names for the dataset.
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEVICE = _select_device()
