import torch

from training.model import Cifar10CNN
from .settings import DEVICE, CHECKPOINT_PATH, MODEL_VERSION

_MODEL = None  # cache the loaded model

def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"CHeckpoint not found: {CHECKPOINT_PATH}")

    model = Cifar10CNN()
    model.to(DEVICE)        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # Handle checkpoints saved with training metadata
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL = model
    return _MODEL


if __name__ == "__main__":
    m = get_model()
    print(f"Model loaded successfully on {DEVICE}. Version={MODEL_VERSION}")
