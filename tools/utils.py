import torch


def load_pretrain(model, pretrained_model):
    ckpt = torch.load(pretrained_model, map_location='cpu')
    state_dict = ckpt['model']

    model_dict = model.state_dict()
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


def load_ckpt(args, model, optimizer, scheduler, logger):
    if args.ckpt is None:
        logger.info('No checkpoint.')
        return

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info(f"=> loaded successfully '{args.checkpoint}' (epoch {checkpoint['epoch']})")
    return checkpoint['epoch']