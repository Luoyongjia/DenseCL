import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import get_backbone


def D(p, z, version='original'):
    if version == 'original':
        z = z.detach()
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_head(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super(projection_head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.num_layers = 2

    def set_layersNum(self, numlayers):
        self.num_layers = numlayers

    def forward(self, x):
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception

        return x


class prediction_head(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super(prediction_head, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class Simsiam(nn.Module):
    def __init__(self, backbone='resnet50', feature_dim=2048, layers_num=2):
        super(Simsiam, self).__init__()
        self.backbone = get_backbone(backbone)
        self.projector = projection_head(self.backbone.output_dim, out_dim=feature_dim)
        self.projector.set_layersNum(layers_num)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.predictor = prediction_head()

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)

        L = D(p1, z2) / 2 + D(p2, z1) / 2

        return {'loss': L}
