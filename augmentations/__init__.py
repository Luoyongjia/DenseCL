from .moco_aug import transform_moco
from .denseCL_aug import transform_denseCL
from .eval_aug import transform_evl


def get_aug(name='denseCL', image_size=224, train=True, train_classifier=None):
    if train:
        if name == 'moco':
            augmentations = transform_moco(image_size)
        elif name == 'denseCL':
            augmentations = transform_denseCL(image_size)
        elif name == 'vapss':
            augmentations = transform_denseCL(image_size)
        else:
            raise Exception

        return augmentations

