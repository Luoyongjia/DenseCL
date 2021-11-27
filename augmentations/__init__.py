from .moco_aug import transform_moco
from .denseCL_aug import transform_denseCL
from .eval_aug import transform_evl
from .single_aug import transform_single


def get_aug(name='denseCL', image_size=224, train=True, train_classifier=None):
    if train:
        if name == 'moco':
            augmentations = transform_moco(image_size)
        elif name == 'denseCL':
            augmentations = transform_denseCL(image_size)
        elif name == 'vapss':
            augmentations = transform_denseCL(image_size)
        else:
            raise NotImplementedError
    else:
        augmentations = transform_single(image_size, train=train_classifier)

    return augmentations

