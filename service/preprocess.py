import torch
import torchvision.transforms as transforms
from PIL import Image
from .settings import MEAN, STD, DEVICE

_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image into a model-ready tensor:
    returns shape [1, 3, 32, 32] on the configured device.
    """
    pil_img = pil_img.convert("RGB")
    img_tensor = _transform(pil_img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim
    img_tensor = img_tensor.to(DEVICE)
    return img_tensor
