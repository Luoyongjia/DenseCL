import os.path

import torch
import torchvision.datasets as datasets


def tinyImageNet(data_dir, transform, train=True):
    """
    ImageFolder: root/dog/xxx.png
                 root/dog/xxy.png

                 root/cat/123.png
                 root/cat/asd.png
    """
    if train:
        dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    else:
        dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)

    return dataset
