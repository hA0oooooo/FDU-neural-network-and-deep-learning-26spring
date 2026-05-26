from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
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


def get_transforms(mean, std, train: bool):
    transforms_list = []
    if train:
        transforms_list.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    transforms_list.append(transforms.ToTensor())
    if mean is not None and std is not None:
        transforms_list.append(transforms.Normalize(mean, std))
    return transforms.Compose(transforms_list)


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _split_indices(length: int, val_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def build_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    val_size: int,
    seed: int,
    mean,
    std,
    train_size: int | None = None,
):
    data_dir = Path(data_dir)
    train_transform = get_transforms(mean, std, train=True)
    eval_transform = get_transforms(mean, std, train=False)

    train_full = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=train_transform
    )
    val_full = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=eval_transform
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=eval_transform
    )

    train_indices, val_indices = _split_indices(len(train_full), val_size, seed)
    if train_size is not None:
        train_indices = train_indices[:train_size]
    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = _make_loader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
    )
    val_loader = _make_loader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = _make_loader(test_set, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def build_cifar10_train_test_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    mean,
    std,
):
    data_dir = Path(data_dir)
    train_set = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=get_transforms(mean, std, train=True),
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=get_transforms(mean, std, train=False),
    )

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = _make_loader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
    )
    test_loader = _make_loader(test_set, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
