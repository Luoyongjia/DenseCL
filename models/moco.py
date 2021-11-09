import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18

from .backbone import resnet18_cifar


def contrastiveLoss(pos, neg, temperature=0.1):
    """
    :param pos(Tensor): Nx1 positive similarity.
    :param neg(Tensor): Nxk negative similarity.

    :return dict[str, Tensor]:  A dictionary of loss components.
    """
    criterion = nn.CrossEntropyLoss()

    N = pos.size(0)
    logits = torch.cat((pos, neg), dim=1)
    logits /= temperature
    labels = torch.zeros((N, ), dtype=torch.long).cuda()
    losses = dict()
    losses['loss_contrastive'] = criterion(logits, labels)

    return losses


class projection_MLP(nn.Module):
    """
    The non-linear neck, fc-relu-fc
    """
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MoCo(nn.Module):
    """
       Build a MoCo model with: a query encoder, a key encoder, and a queue
   """
    def __init__(self, backbone=resnet50()):
        super(MoCo, self).__init__()

        # K: queue size, m: momentum of updating keys, T: softmax temperature, dim: feature dim
        self.K = 65536
        self.m = 0.999
        self.T = 0.07
        self.dim = backbone.output_dim

        # mpl: whether using mlp head
        self.mlp = False

        # create the encoders
        self.encoder_q = backbone
        self.encoder_k = backbone

        if self.mlp:
            self.encoder_q.fc = nn.Sequential(projection_MLP(self.dim), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(projection_MLP(self.dim), self.encoder_k.fc)

        # initial param of encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """update the queue."""
        # gather keys before updating queue.
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr(dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K   # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm
        Only support DistributedDataParallel(DDP) model.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rand()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, img_k):
        """

        :param img_q(Tensor): Input of a batch of query images (N, C, H, W)
        :param img_k: Input of a batch of key images (N, C, H, W)
        :return: dict[str, Tensor]: A dictionary of loss components.
        """

        # compute query feature
        q = self.encoder_q(img_q)
        q = F.normalize(q, dim=1)

        # compute the key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(im_k)[0]     # keys: [N, C]
            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nv, ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        losses = contrastiveLoss(l_pos, l_neg, temperature=self.T)

        self._dequeue_and_enqueue(k)
        return losses


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
