from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import MEAN, STD, BATCH_SIZE

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(MEAN, STD,)]
)

train_dataset = datasets.CIFAR10("./data", 
                                 train=True,
                                 download=True,
                                 transform=transform)

test_dataset = datasets.CIFAR10("./data",
                                train=False,
                                download=False,
                                transform=transform)

def get_dataloaders(train_dataset, test_dataset, BATCH_SIZE):
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    
    test_dataloader = DataLoader(test_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, test_dataloader = get_dataloaders(train_dataset, test_dataset, BATCH_SIZE)
    train_iter = iter(train_dataloader)
    images, labels = next(train_iter)
    print(f"Images Tensor Shape: {images.shape}")
    print(f"Labels Tensor Shape: {labels.shape}")