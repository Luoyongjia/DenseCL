import os
from tools import get_args


args = get_args()

if not os.path.exists(f'./res/{args.name}/{args.exp_num}'):
    os.mkdir(f'./res/{args.name}/{args.exp_num}')
if not os.path.exists(f'./res/{args.exp_num}'):
    os.mkdir(f'./res/{args.exp_num}')
if not os.path.exists(f'./res/{args.exp_num}/checkpoints'):
    os.mkdir(f'./res/{args.exp_num}/checkpoints')
if not os.path.exists(f'./res/{args.exp_num}/logs'):
    os.mkdir(f'./res/{args.exp_num}/logs')