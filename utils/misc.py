import os
import torch


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model checkpoint to disk.

    Args:
        state (dict): Information to be saved in the checkpoint, including model state dict, optimizer state dict, etc.
        is_best (bool): Whether this checkpoint is the best one so far.
        checkpoint_dir (str): Directory to save the checkpoint.

    Returns:
        None

    """
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_path)


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar'):
    """Loads model checkpoint from disk.

    Args:
        model (torch.nn.Module): Model to load the checkpoint to.
        optimizer (torch.optim.Optimizer): Optimizer to load checkpoint to, if specified.
        filename (str): Checkpoint filename.

    Returns:
        int: Start epoch to continue training.

    """
    start_epoch = 0
    if os.path.isfile(filename):
        print("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("No checkpoint found at '{}'".format(filename))

    return start_epoch


"""
该脚本主要提供了两个函数，用于模型训练过程中保存和加载模型的状态。

save_checkpoint函数接收三个参数：

state：一个字典，包含了需要保存到文件中的信息，如模型参数、优化器参数等。
is_best：一个布尔值，表示这个状态是否是目前最佳的状态。
checkpoint_dir：表示将保存状态的目录。
该函数将状态信息保存在checkpoint_dir目录下，文件名为checkpoint.pth.tar，如果is_best为True，则另外再保存一份文件，文件名为model_best.pth.tar。

load_checkpoint函数用于加载保存的模型状态。接收三个参数：

model：需要加载状态的模型。
optimizer：优化器，如果需要将优化器的状态一并加载，可以传入此参数。
filename：状态所在的文件名。
该函数会尝试从指定的文件名中加载状态，如果找到该文件，则会返回上一次训练的开始 epoch，同时将状态加载到指定的模型和优化器中。如果找不到文件，则返回默认的开始 epoch 0。
"""