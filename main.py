import os
import torch
# import torch.distributed as dist

from tools import get_args, Logger, writer, train
from datasets import get_dataset
from augmentations import get_aug
from models import get_model
from optimizers import get_optimizer, LR_Scheduler
from evaluation import linear_eval


def main(args):
    # # setting singl GPU
    # print('==>Changing GPU.')
    # dist.init_process_group('nccl', init_method='file:///temp/somefile', rank=0, world_size=1)
    # build the ./res
    print('==> Creating dices.')
    if not os.path.exists(f'./res'):
        os.mkdir(f'./res')
    if not os.path.exists(f'./res/{args.name}'):
        os.mkdir(f'./res/{args.name}')
    if not os.path.exists(f'./res/{args.name}/{args.exp_num}'):
        os.mkdir(f'./res/{args.name}/{args.exp_num}')
    if not os.path.exists(f'./res/{args.name}/{args.exp_num}/checkpoints'):
        os.mkdir(f'./res/{args.name}/{args.exp_num}/checkpoints')
    if not os.path.exists(f'./res/{args.name}/{args.exp_num}/logs'):
        os.mkdir(f'./res/{args.name}/{args.exp_num}/logs')
    if not os.path.exists(f'./res/{args.name}/{args.exp_num}/logs/tensorboard'):
        os.mkdir(f'./res/{args.name}/{args.exp_num}/logs/tensorboard')

    logPath = f'./res/{args.name}/{args.exp_num}/logs/'
    writerPath = f'./res/{args.name}/{args.exp_num}/logs/tensorboard'

    logger = Logger(f'{args.name}/{args.exp_num}', logPath)
    boardwriter = writer(log_dir=writerPath)

    # load data
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs
        ),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    linear_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.eval.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs
        ),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # build model
    model = get_model(args.model).to(args.device)
    model = torch.nn.DataParallel(model)

    optimizer = get_optimizer(
        args.train.optimizer.name, model,
        lr=args.train.base_lr,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs,
        args.train.warmup_lr,
        args.train.epochs,
        args.train.base_lr,
        args.train.final_lr,
        len(train_loader),
        constant_predictor_lr=False      # simsiam section 4.2 predictor
    )

    train(train_loader, memory_loader, test_loader, model, optimizer, lr_scheduler, logger, boardwriter, args)
    ckptPath = f'./res/{args.name}/{args.exp_num}/checkpoints/ckpt-last.pth'

    linear_eval(linear_loader, test_loader, ckptPath, logger, args)


if __name__ == "__main__":
    args = get_args()
    main(args)