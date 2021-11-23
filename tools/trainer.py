import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
from tqdm import tqdm
import time

from evaluation import knn_monitor


def load_checkpoint(ckpt_path, model):
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        return checkpoint['epoch'] - 1
    else:
        return 0


def train(train_loader, memo_loader, test_loader, model, optimizer, scheduler, logger, writer, args):
    epoch_pre = load_checkpoint(model, args.checkpoint)
    args.train.epoches = args.train.epoches - epoch_pre

    accuracy = 0
    start_time = time.time()

    # start training
    global_progress = tqdm(range(0, args.train.epoches), desc='Training')
    for epoch in global_progress:
        model.train()

        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.epoches}')
        for idx, ((img_q, img_k), labels) in enumerate(local_progress):
            model.zerp_grad()
            data_dict = model.forward(img_q, img_k)
            loss = data_dict['loss_contrastive'].mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            data_dict.update({'lr': scheduler.get_lr()})

            local_progress.set_postfix(data_dict)
            writer.update_scalers(data_dict)

        if epoch % args.evl_frequent == 0:
            accuracy = knn_monitor(model.module.backbone, memo_loader, test_loader,
                                   k=min(args.train.knn_k, len(memo_loader.dataset)))
            logger.info(f'[{epoch} / {args.train.epoches}]: accuracy: {accuracy}, lr:{scheduler.get_lr()}')
        epoch_dict = {"epoch": epoch, "accuracy": accuracy}
        global_progress.set_postfix(epoch_dict)
        writer.update_scalers(epoch_dict)

        # save checkpoints
        ckpt_path = os.path.join(args.ckpt_path, f'{args.exp_num}/checkpoints/ckpt-{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)

    # save model
    model_path = os.path.join(args.ckpt_path, f'{args.exp_num}/checkpoints/ckpt-last.pth')
    torch.save({
        'epoch': args.train.epoches,
        'model': model.state_dict()
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Spent {((time.time() - start_time)/3600):.4f} h")

