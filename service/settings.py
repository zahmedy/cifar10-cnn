from pathlib import Path
import torch


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Inference device
DEVICE = _select_device()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# CIFAR-10 class names (order matters)
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

# Model checkpoint (one level above this folder)
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cifar10_cnn.pt"

# Human-readable version string
MODEL_VERSION = "cifar10-cnn-v01"

# Normalization stats for CIFAR-10
# These numbers center/scale images so training is stable.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)