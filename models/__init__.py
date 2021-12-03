from .moco import MoCo
from .denseCL import DenseCL
from .simsiam import Simsiam
from .backbones import get_backbone
import torch
# from torchvision.models import resnet50, resnet18
from .backbones import resnet18_cifar, resnet18, resnet50


def get_model(cfg_model):
    if cfg_model.name == 'ours':
        model = MoCo(cfg_model.backbone,
                     feature_dim=cfg_model.feature_dim,
                     K=cfg_model.hyperparameter.K,
                     m=cfg_model.hyperparameter.m,
                     T=cfg_model.hyperparameter.T)
    elif cfg_model.name == 'moco':
        model = MoCo(cfg_model.backbone,
                     feature_dim=cfg_model.feature_dim,
                     K=cfg_model.hyperparameter.K,
                     m=cfg_model.hyperparameter.m,
                     T=cfg_model.hyperparameter.T)
    elif cfg_model.name == 'denseCL':
        model = DenseCL(cfg_model.backbone,
                        feature_dim=cfg_model.feature_dim,
                        K=cfg_model.hyperparameter.K,
                        m=cfg_model.hyperparameter.m,
                        T=cfg_model.hyperparameter.T
                        )
    elif cfg_model.name == 'simsiam':
        model = Simsiam(cfg_model.backbone,
                        feature_dim=cfg_model.feature_dim,
                        layers_num=cfg_model.layers_num
                        )

    else:
        raise NotImplementedError

    return model

