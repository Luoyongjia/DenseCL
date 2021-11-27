import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50, resnet18

from .backbones import resnet18_cifar, resnet50, resnet18


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
    losses = criterion(logits, labels)

    return losses


class projection_conv(nn.Module):
    """
    A non-linear neck in DenseCL
    The non-linear neck, fc-relu-fc, conv-relu-conv
    """
    def __init__(self, in_dim, hid_dim=2048, out_dim=128, s=None):
        super(projection_conv, self).__init__()
        self.is_s = s
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hid_dim, out_dim))
        self.mlp_conv = nn.Sequential(nn.Conv2d(in_dim, hid_dim, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(hid_dim, out_dim, 1))
        if self.is_s:
            self.pool = nn.AdaptiveAvgPool2d((s, s))
        else:
            self.pool = None

    def forward(self, x):
        # Global feature vector
        x1 = self.avgpool(x)
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.mlp(x1)

        # dense feature map
        if self.is_s:
            x = self.pool(x)                        # [N, C, S, S]
        x2 = self.mlp_conv(x)
        x2 = x2.view(x2.size(0), x2.size(1), -1)    # [N, C, SxS]

        x3 = self.avgpool(x2)                       # [N, C, 1, 1]
        x3 = x3.view(x3.size(0), -1)                # [N, C]
        return x, x1, x2, x3


class DenseCL(nn.Module):
    """
       Build a MoCo model with: a query encoder, a key encoder, and a queue
       num_grid: divided into s x s grid.
   """
    def __init__(self, backbone=resnet50(), feature_dim=128, K=65536, m=0.999, T=0.07, loss_lambda=0.5, num_grid=3):
        super(DenseCL, self).__init__()

        # K: queue size, m: momentum of updating keys, T: softmax temperature, dim: feature dim
        self.K = K
        self.m = m
        self.T = T
        self.loss_lambda = loss_lambda
        self.dim = backbone.output_dim
        self.out_dim = feature_dim

        # create the encoders
        # self.encoder_q = backbone
        # self.encoder_k = backbone
        #
        # self.encoder_q.fc = nn.Sequential(projection_conv(in_dim=self.dim, out_dim=self.out_dim, s=num_grid), self.encoder_q.fc)
        # self.encoder_k.fc = nn.Sequential(projection_conv(in_dim=self.dim, out_dim=self.out_dim, s=num_grid), self.encoder_k.fc)

        self.backbone_q = backbone
        self.projector_q = projection_conv(in_dim=self.dim, out_dim=self.out_dim, s=num_grid)
        self.backbone_k = backbone
        self.projector_k = projection_conv(in_dim=self.dim, out_dim=self.out_dim, s=num_grid)

        self.encoder_q = nn.Sequential(
            self.backbone_q,
            self.projector_q
        )
        self.encoder_k = nn.Sequential(
            self.backbone_k,
            self.projector_k
        )

        # initial param of encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.out_dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_dense", torch.randn(self.out_dim, self.K))
        self.queue_dense = F.normalize(self.queue_dense, dim=0)

        # queue init.
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_dense_ptr", torch.zeros(1, dtype=torch.long))

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
    def _dequeue_and_enqueue_dense(self, keys):
        """update the queue."""
        # gather keys before updating queue.
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_dense_ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr(dequeue and enqueue)
        self.queue_dense[:, ptr: ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_dense_ptr[0] = ptr

    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     """
    #     Batch shuffle, for making use of BatchNorm
    #     Only support DistributedDataParallel(DDP) model.
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()
    #
    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)
    #
    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)
    #
    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rand()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this], idx_unshuffle
    #
    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]
    #
    #     num_gpus = batch_size_all // batch_size_this
    #
    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
    #
    #     return x_gather[idx_this]

    def forward(self, img_q, img_k):
        """

        :param img_q(Tensor): Input of a batch of query images (N, C, H, W)
        :param img_k: Input of a batch of key images (N, C, H, W)
        :return: dict[str, Tensor]: A dictionary of loss components.
        """
        # compute query feature
        # input [N, C, H, W]
        q_test = self.encoder_q[0](img_q)
        q_orig, q, q_grid, q2 = self.encoder_q[1](q_test)

        q_orig = nn.functional.normalize(q_orig, dim=1)
        q = nn.functional.normalize(q, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        # compute the key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            # img_k, idx_unshffle = self._batch_shuffle_ddp(img_k)

            k_orig, k, k_grid, k2 = self.encoder_k(img_k)     # keys: [N, C], [N, C, SxS]

            k_orig = nn.functional.normalize(k_orig, dim=1)
            k = nn.functional.normalize(k, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)

            # # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshffle)
            # k2 = self._batch_shuffle_ddp(k2, idx_unshffle)
            # k_grid = self._batch_shuffle_ddp(k_grid, idx_unshffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # compute dense loss of backbone
        # q_grid: [N, SxS, C], k_grid: [N, C, SxS], backbone similarity matrix: [N, SxS,SxS]
        backbone_sim_matrix = torch.matmul(q_grid.permute(0, 2, 1), k_grid)
        # chose the most similar index
        densecl_sim_index = backbone_sim_matrix.max(dim=2)[1]
        # get the positive feature vector from k_grid, [N, C, SxS]
        indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_index.unsqueeze(1).expand(-1, k_grid.size(1), -1))
        # calculating the pos pair, [N, SxS]
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1)

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))

        # pos samples: [NxSxS, 1]
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)
        # neg samples: [NxSxS, k]
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid, self.queue_dense.clone().detach()])

        loss_contra_single = contrastiveLoss(l_pos, l_neg, temperature=self.T)
        loss_contra_dense = contrastiveLoss(l_pos_dense, l_neg_dense, temperature=self.T)
        loss_contrastive = (1 - self.loss_lambda) * loss_contra_single + self.loss_lambda * loss_contra_dense
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_dense(k2)

        # loss_contrastive = loss_contrastive.requires_grad_()
        return {
                'loss': loss_contrastive,
                'loss_contra_single': loss_contra_single,
                'loss_contra_dense': loss_contra_dense
                }


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor)
    #                   for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    tensors_gather = [torch.ones_like(tensor)]

    output = torch.cat(tensors_gather, dim=0)
    return output

