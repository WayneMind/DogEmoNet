import os
import torch


def save_checkpoint(state, is_best, checkpoint_dir):
    """
    保存模型参数，保存最佳模型参数。
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best.pth.tar')
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(checkpoint_dir, model, optimizer=None):
    """
    加载模型参数，可选择加载优化器参数。
    """
    filepath = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    if not os.path.exists(filepath):
        raise RuntimeError("Checkpoint '{}' not found".format(filepath))
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


"""
save_checkpoint()函数用于保存模型参数，其中state是一个字典类型的对象，包含了模型、优化器以及相关超参数的状态信息。is_best表示该模型是否是最佳模型，如果是，则会将其保存在checkpoint_dir目录下的best.pth.tar文件中，否则只会保存在checkpoint_dir目录下的checkpoint.pth.tar文件中。

load_checkpoint()函数用于加载模型参数，其中checkpoint_dir表示存储模型参数的目录，model是已经定义好的模型，optimizer是优化器（如果有）。该函数返回checkpoint，其中包含了模型参数和优化器参数。

在训练模型时，我们通常需要在每个epoch结束时保存模型参数，以便以后加载和使用。通过调用save_checkpoint()函数，可以将当前模型的参数保存到文件中。

在使用训练好的模型进行测试时，我们通常需要先加载训练好的模型参数，然后再使用该模型进行测试。通过调用load_checkpoint()函数，可以从之前训练好的模型参数文件中加载模型参数，并将其赋值给已经定义好的模型。
"""