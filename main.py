import os
import torch


from tools import get_args
from datasets import get_dataset
from augmentations import get_aug


def main(args):
    # build the ./res
    if not os.path.exists(f'./res/{args.name}'):
        os.mkdir(f'./res/{args.name}')
    if not os.path.exists(f'./res/{args.name}/{args.exp_num}'):
        os.mkdir(f'./res/{args.name}/{args.exp_num}')
    if not os.path.exists(f'./res/{args.name}/{args.exp_num}/checkpoints'):
        os.mkdir(f'./res/{args.name}/{args.exp_num}/checkpoints')
    if not os.path.exists(f'./res/{args.name}/{args.exp_num}/logs'):
        os.mkdir(f'./res/{args.name}/{args.exp_num}/logs')

    # load data
    train_loader = torch.utils.data.DataLoader(
        datasets=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs
        ),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )


if __name__ == "__main__":
    args = get_args()

    main(args)