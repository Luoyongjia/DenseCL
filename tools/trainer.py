import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
from tqdm import tqdm
import time

from evaluation import knn_monitor


# def load_checkpoint(ckpt_path, model, logger):
#     if ckpt_path != 'None':
#         logger.info(f'Loading checkpoint from {ckpt_path}.')
#         checkpoint = torch.load(ckpt_path, map_location="cpu")
#         model.load_state_dict(checkpoint['model'])
#         return checkpoint['epoch'] - 1
#     else:
#         return 0


def train(train_loader, memo_loader, test_loader, model, optimizer, scheduler, logger, writer, args):
    logger.info(f'====>Training.')
    logger.info(f'====>Using device {args.device}.')
    print(f'====>Training.')
    print(f'====>Using device {args.device}.')

    # epoch_pre = load_checkpoint(args.checkpoint, model, logger)
    # args.train.epochs = args.train.epochs - epoch_pre
    logger.info(f'====>Training {args.train.epochs} epoch. ')
    print(f'====>Training {args.train.epochs} epoch. ')

    accuracy = 0
    start_time = time.time()

    # start training
    global_progress = tqdm(range(0, args.train.epochs), desc='Training')
    for epoch in global_progress:
        model.train()

        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.epochs}')
        for idx, ((img_q, img_k), labels) in enumerate(local_progress):
            model.zero_grad()
            data_dict = model.forward(img_q, img_k)
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            data_dict.update({'lr': scheduler.get_lr()})

            local_progress.set_postfix({"loss": data_dict['loss'].item(),
                                        "loss_contra": data_dict['loss_contra_single'].item(),
                                        "loss_contra_dense": data_dict['loss_contra_dense'].item()})
            writer.update_scalers(data_dict)

        if epoch % args.evl_frequent == 0:
            accuracy = knn_monitor(model.module.backbone_q, memo_loader, test_loader,
                                   k=min(args.train.knn_k, len(memo_loader.dataset)))
            logger.info(f'[{epoch} / {args.train.epochs}]: accuracy: {accuracy}, lr:{scheduler.get_lr()}')
        epoch_dict = {"epoch": epoch, "accuracy": accuracy}
        global_progress.set_postfix(epoch_dict)
        writer.update_scalers(epoch_dict)

        # save checkpoints
        ckpt_path = os.path.join(args.res_dir, f'{args.name}/{args.exp_num}/checkpoints/ckpt-{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
        }, ckpt_path)

    # save model
    model_path = os.path.join(args.res_dir, f'{args.name}/{args.exp_num}/checkpoints/ckpt-last.pth')
    torch.save({
        'epoch': args.train.epochs,
        'model': model.state_dict()
    }, model_path)
    logger.info(f"====>Model saved to {model_path}")
    logger.info(f"====>Spent {((time.time() - start_time)/3600):.4f} h")
    print(f"====>Model saved to {model_path}")
    print(f"====>Spent {((time.time() - start_time) / 3600):.4f} h")

