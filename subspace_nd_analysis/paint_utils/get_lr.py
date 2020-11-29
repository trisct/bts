def get_lr(optimizer):
    for params in optimizer.param_groups:
        return params['lr']