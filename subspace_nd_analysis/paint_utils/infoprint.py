def print_training_info(opt):
    print('--------------Training Info--------------')
    print('Number of epochs in total: %d'%opt.epochs)
    print('Learning rate: %f'%opt.lr)
    print('Loss function: MSELoss')
    print('Optimizer: Adam')
    print('Scheduler: StepLR')
    print('Scheduler step size: %d'%opt.scheduler_step_size)
    print('Scheduler gamma: %f'%opt.scheduler_gamma)
    print('-------------------End-------------------')