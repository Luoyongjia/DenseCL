import argparse
import yaml
import shutil
import re

import yaml


class Namespace(object):
    """
    load the configs from xxxx.yaml
    args.key
    """

    def __init__(self, dic):
        for key, value in dic.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in config file!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="./configs/denseCL_cifar10.yaml", help="config film, xxxx.yaml")

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    if args.debug:
        if args.train:
            args.train.batch_size = 2
            args.train.epochs = 1
        if args.eval:
            args.eval.batch_size = 2
            # retrain 1 epoch
            args.eval.epochs = 1
        args.dataset.num_workers = 0

    vars(args)['aug_kwargs'] = {
        'name': args.model.name,
        'image_size': args.data_dir,
    }
    vars(args)['dataset_kwargs'] = {
        'dataset': args.dataset.name,
        'data_dir': args.dataset.data_dir,
        'download': args.download,
        'debug_subset_size': args.debug.subset_size if args.debug else None,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers
    }

    return args

