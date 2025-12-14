import torch

from model import Cifar10CNN
from .settings import DEVICE, CHECKPOINT_PATH, MODEL_VERSION, NUM_CLASSES

_MODEL = None  # cache the loaded model


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = Cifar10CNN(NUM_CLASSES)
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            # Handle checkpoints saved with training metadata
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        except Exception:
            print("Unable to load from checkpoint")
            raise
        _MODEL.load_state_dict(state_dict)
        _MODEL.eval()
    return _MODEL


if __name__ == "__main__":
    model = get_model()
    print("Model loaded successfully!")
