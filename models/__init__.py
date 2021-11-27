from .moco import MoCo
from .denseCL import DenseCL

import torch
# from torchvision.models import resnet50, resnet18
from .backbones import resnet18_cifar, resnet18, resnet50


def get_backbone(backbone, castrate=True):
    backbone = eval(f'{backbone}()')

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(cfg_model):
    if cfg_model.name == 'ours':
        model = MoCo(get_backbone(cfg_model.backbone),
                     feature_dim=cfg_model.feature_dim,
                     K=cfg_model.hyperparameter.K,
                     m=cfg_model.hyperparameter.m,
                     T=cfg_model.hyperparameter.T)
    elif cfg_model.name == 'moco':
        model = MoCo(get_backbone(cfg_model.backbone),
                     feature_dim=cfg_model.feature_dim,
                     K=cfg_model.hyperparameter.K,
                     m=cfg_model.hyperparameter.m,
                     T=cfg_model.hyperparameter.T)
    elif cfg_model.name == 'denseCL':
        model = DenseCL(get_backbone(cfg_model.backbone),
                        feature_dim=cfg_model.feature_dim,
                        K=cfg_model.hyperparameter.K,
                        m=cfg_model.hyperparameter.m,
                        T=cfg_model.hyperparameter.T
                        )

    else:
        raise NotImplementedError

    return model

