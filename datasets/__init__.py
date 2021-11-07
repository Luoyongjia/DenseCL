import torch
import torchvision

from .build_datasets import BuildDataset


def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):
    if dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, train=train, transform=transform, download=download)
    else:
        dataset = BuildDataset()

    if debug_subset_size is not None:
        # for debug, only take one batch
        dataset = torch.utils.dataset.Subset(dataset, range(0, debug_subset_size))
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets

    return dataset
