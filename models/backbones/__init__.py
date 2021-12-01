import torch
from .Resnet_cifar import resnet18_cifar
from .Resnet import resnet18, resnet50


def get_backbone(backbone, castrate=True):
    backbone = eval(f'{backbone}()')

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone
