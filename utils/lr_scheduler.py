import math

import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, args):
    """返回学习率调度器

    Args:
        optimizer (torch.optim.Optimizer): 优化器
        args (argparse.Namespace): 命令行参数

    Returns:
        torch.optim.lr_scheduler._LRScheduler: 学习率调度器
    """
    if args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy {} is not implemented'.format(args.lr_policy))
    return scheduler


"""
该函数返回一个 PyTorch 学习率调度器对象，用于调整优化器的学习率。学习率调度器可以根据训练过程中的一些指标来动态地调整学习率，以提高训练效果。

该函数根据命令行参数 args 中的 lr_policy 参数返回不同的学习率调度器：

如果 args.lr_policy 为 step，则返回一个 StepLR 学习率调度器。该调度器每隔 args.lr_decay_iters 个 epoch 将学习率乘以 0.1。
如果 args.lr_policy 为 cosine，则返回一个 CosineAnnealingLR 学习率调度器。该调度器按照余弦函数调整学习率，使学习率在训练期间从初始值下降到 0，训练的总轮数为 args.num_epochs。
如果 args.lr_policy 不是 step 或 cosine，则会抛出 NotImplementedError 异常，提示该学习率调度策略未实现。
注释中给出了函数的输入和输出以及每个参数的作用。函数实现了两种不同的学习率调度策略，分别是 StepLR 和 CosineAnnealingLR。其中 StepLR 每隔一定轮数就将学习率降低，适合于逐步降低学习率的场景。而 CosineAnnealingLR 是通过余弦函数逐步降低学习率，适合于更平滑的降低学习率。
"""