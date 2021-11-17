from .moco import MoCo
from .denseCL import DenseCL

import torch
from torchvision.models import resnet50, resnet18
from .backbones import resnet18_cifar


def get_backbone(backbone, feature_dim=128, castrate=True):
    backbone = eval('{}(num_classes={})'.format(backbone, feature_dim))

    if castrate:
        backbone.output_dim = backbone.fc.in_feature
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(cfg_model):
    if cfg_model.name == 'ours':
        model = MoCo(get_backbone(cfg_model.backbone, cfg_model.feature_dim),
                     K=cfg_model.hyperparameter.K,
                     m=cfg_model.hyperparameter.m,
                     T=cfg_model.hyperparameter.T)
    elif cfg_model.name == 'moco':
        model = MoCo(get_backbone(cfg_model.backbone, cfg_model.feature_dim),
                     K=cfg_model.hyperparameter.K,
                     m=cfg_model.hyperparameter.m,
                     T=cfg_model.hyperparameter.T)
    elif cfg_model.name == 'densecl':
        model = DenseCL(get_backbone(cfg_model.name, cfg_model.feature_dim),
                        K=cfg_model.hyperparameter.K,
                        m=cfg_model.hyperparameter.m,
                        T=cfg_model.hyperparameter.T
                        )

    else:
        raise NotImplementedError

    return model

