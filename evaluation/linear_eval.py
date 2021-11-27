import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from optimizers import LR_Scheduler
from models import *
from tools import AverageMeter


def linear_eval(train_loader, test_loader, ckpt_path, logger, args):
    checkpoint = torch.load(ckpt_path)
    model = get_backbone(args.model.backbone)
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    classifier = nn.Linear(in_features=model.output_dim, out_features=10, bias=True).to(args.device)
    # assert args.eval_from is not None
    save_dict = checkpoint['model']
    msg = model.load_state_dict({k[len('module.backbone_q.'):]: v for k, v in save_dict.items() if k.startswith('module.backbone_q.')},
                                strict=True)

    model = model.to(args.device)
    model = nn.DataParallel(model)

    # classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = nn.DataParallel(classifier)

    optimizer = torch.optim.SGD(classifier.parameters(),
                                lr=args.eval.base_lr,
                                momentum=args.eval.optimizer.momentum,
                                weight_decay=args.eval.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs,
        args.eval.warmup_lr,
        args.eval.epochs,
        args.eval.base_lr,
        args.eval.final_lr,
        len(train_loader),
    )

    loss_meter = AverageMeter(name='Loss')
    # training
    global_progress = tqdm(range(0, args.eval.epochs), desc='Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.epochs}', disable=True)

        for idx, (image, labels) in enumerate(local_progress):
            classifier.zero_grad()
            with torch.no_grad():
                feature = model(image.to(args.device))
            # for feature map output
            feature = avgpool(feature)
            feature = feature.reshape(feature.size(0), -1)

            preds = classifier(feature)

            loss = F.cross_entropy(preds, labels.to(args.device))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            local_progress.set_postfix({'lr': lr, 'loss': loss_meter.val, 'loss_avg': loss_meter.avg})

    classifier.eval()
    correct, total = 0, 0
    acc1_meter = AverageMeter(name='Acc@1')
    # acc5_meter = AverageMeter(name='Acc@5')
    for idx,  (image, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model(image.to(args.device))
            # for feature map output
            feature = avgpool(feature)
            feature = feature.reshape(feature.size(0), -1)

            preds = classifier(feature).argmax(dim=1)
            acc1 =(preds == labels.to(args.device)).sum().item()
            acc1_meter.update(acc1/preds.shape[0])
            # acc5_meter.update(acc5[0], preds.shape[0])
    # logger.info(f'Acc@1 = {acc1_meter.avg * 100: .4f}, Acc@5 = {acc5_meter.avg * 100: .4f}.')
    logger.info(f'Acc@1 = {acc1_meter.avg * 100: .4f}')
