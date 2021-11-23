import os
import torch

from tools import get_args, Logger, writer
from datasets import get_dataset
from augmentations import get_aug
from models import get_model


def main(args):
    # build the ./res
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

    ckptPath = f'./res/{args.name}/{args.exp_num}/'
    logPath = f'./res/{args.name}/{args.exp_num}/logs'

    logger = Logger('test', logPath)
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


if __name__ == "__main__":
    args = get_args()
    main(args)