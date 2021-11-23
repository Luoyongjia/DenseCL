import torch
import torchvision

from .build_datasets import LoadDataset
from .tinyImageNet import tinyImageNet

import augmentations


def get_dataset(name, data_dir, transform, train=True, download=False, debug_subset_size=None):
    if name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif name == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, train=train, transform=transform, download=download)
    elif name == 'tiny-imagenet':
        dataset = tinyImageNet(data_dir, train=train, transform=transform)
    else:
        dataset = LoadDataset(data_dir)

    if debug_subset_size is not None:
        # for debug, only take one batch
        dataset = torch.utils.dataset.Subset(dataset, range(0, debug_subset_size))
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets

    return dataset


# if __name__ == "__main__":
#     tinyImageNet_train = get_dataset(dataset='tiny-imagenet',
#                                      data_dir='/Users/luoyongjia/Research/Data/tiny-imagenet-200',
#                                      transform=augmentations.get_aug(image_size=32),
#                                      train=True,)
