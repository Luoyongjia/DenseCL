import torch

from .lr_scheduler import LR_Scheduler


def get_optimizer(name, model, lr, momentum, weight_decay):
    predictor_prefix = ('module.projector_q',
                        'predictor_q')
    parameters = [
        {
            'name': 'base',
            'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
            'lr': lr
        },
        {
            'name': 'predictor',
            'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
            'lr': lr
        }]

    if name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer
