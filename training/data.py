from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from training.config import (
    BATCH_SIZE,
    DATA_DIR,
    MEAN,
    NUM_WORKERS,
    SEED,
    STD,
    VAL_SPLIT,
)


def _build_transforms(train: bool = True) -> transforms.Compose:
    # Data transforms clean/standardize images; training uses light augmentation.
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def get_datasets() -> Tuple[datasets.CIFAR10, datasets.CIFAR10, datasets.CIFAR10]:
    # Download CIFAR-10 and split the original training set into train/validation.
    train_full = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=True,
        transform=_build_transforms(train=True),
    )
    test_dataset = datasets.CIFAR10(
        DATA_DIR,
        train=False,
        download=True,
        transform=_build_transforms(train=False),
    )

    val_size = int(len(train_full) * VAL_SPLIT)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size: int = BATCH_SIZE):
    # Wrap datasets in DataLoader objects so batches stream efficiently during training.
    train_dataset, val_dataset, test_dataset = get_datasets()
    common_args = {
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_args)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    sample_images, sample_labels = next(iter(train_loader))
    print(f"Train batch: {sample_images.shape}, labels shape: {sample_labels.shape}")
    print(f"Val size: {len(val_loader.dataset)} | Test size: {len(test_loader.dataset)}")
