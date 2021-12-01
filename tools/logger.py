import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from collections import OrderedDict
import logging
import os
import time
import matplotlib

matplotlib.use('Agg')


class Plotter(object):
    def __init__(self):
        self.logger = OrderedDict()

    def update(self, ordered_dict):
        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.logger.get(key) is None:
                self.logger[key] = [value]
            else:
                self.logger[key].append(value)

    def save(self, file, **kwargs):
        fig, axes = plt.subplots(nrows=len(self.logger), ncols=1, figsize=(8, 2*len(self.logger)))
        fig.tight_layout()
        for ax, (key, value) in zip(axes, self.logger.items()):
            ax.plot(value)
            ax.set_title(key)

        plt.savefig(file, **kwargs)
        plt.close()


class writer(object):
    def __init__(self, log_dir, tensorboard=True, matplotlib=True):
        self.reset(log_dir=log_dir, tensorboard=tensorboard, matplotlib=matplotlib)

    def reset(self, log_dir=None, tensorboard=True, matplotlib=True):
        if log_dir is not None:
            self.log_dir = log_dir

        if tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        if matplotlib:
            self.plotter = Plotter()
        else:
            self.plotter = None

        self.counter = OrderedDict()

    def update_scalers(self, ordered_dict):
        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1

            if self.writer:
                self.writer.add_scalar(key, value, self.counter[key])

        if self.plotter:
            self.plotter.update(ordered_dict)
            self.plotter.save(os.path.join(self.log_dir, 'plotter.svg'))


def Logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))[:-4]

    logName = path + rq + '.log'
    fh = logging.FileHandler(logName, mode='a')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


