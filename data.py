from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision


class DataFactory:
    # This factory contains methods to create dataloaders for different datasets
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_mnist_loaders(
        self, batch_size: int, data_path: str = "~/.pytorch/MNIST_data/"
    ) -> tuple[DataLoader, DataLoader]:
        """Creates dataloaders for MNIST dataset

        Args:
            batch_size (int): size of batch
            data_path (str, optional): path for loading . Defaults to "~/.pytorch/MNIST_data/".

        Returns:
            tuple: trainloader, testloader
        """
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST(
            data_path, download=True, train=True, transform=transform
        )
        testset = datasets.MNIST(
            data_path, download=True, train=False, transform=transform
        )
        return self.create_loaders(trainset, testset, batch_size=batch_size)

    def get_fashion_mnist_loaders(
        self, batch_size: int, data_path: str = "~/.pytorch/FashionMNIST_data/"
    ) -> tuple[DataLoader, DataLoader]:
        """Creates dataloaders for FashionMNIST dataset

        Args:
            batch_size (int): batch size
            data_path (str, optional): path for loading. Defaults to "~/.pytorch/FashionMNIST_data/".

        Returns:
            tuple: trainloader, testloader
        """
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.FashionMNIST(
            data_path, download=True, train=True, transform=transform
        )
        testset = datasets.FashionMNIST(
            data_path, download=True, train=False, transform=transform
        )
        return self.create_loaders(trainset, testset, batch_size=batch_size)

    def get_svhn_loaders(
        self, batch_size: int, data_path: str = "~/.pytorch/svhn_data/"
    ) -> tuple[DataLoader, DataLoader]:
        """Creates dataloaders for SVHN dataset

        Args:
            batch_size (int): batch size
            data_path (str, optional): path for loading. Defaults to "~/.pytorch/svhn_data/".

        Returns:
            tuple: trainloader, testloader
        """
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        trainset = torchvision.datasets.SVHN(
            data_path, split="train", download=True, transform=transform
        )
        testset = torchvision.datasets.SVHN(
            data_path, split="test", download=True, transform=transform
        )
        return self.create_loaders(trainset, testset, batch_size=batch_size)

    def create_loaders(
        self, trainset: Dataset, testset: Dataset, batch_size: int = None
    ) -> tuple[DataLoader, DataLoader]:
        """Creates dataloaders for train and test sets

        Args:
            trainset (Dataset): Dataset for training
            testset (Dataset): Dataset for testing
            batch_size (int, optional): batch size. Defaults to None.

        Returns:
            tuple: trainloader, testloader
        """
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        return trainloader, testloader
