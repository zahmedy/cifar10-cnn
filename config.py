
from pathlib import Path
import torch

def _select_device() -> torch.device:
    """Prefer CUDA, then MPS, and finally CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_CLASSES = 10

# Normalization stats for CIFAR-10
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

DEVICE = _select_device()
