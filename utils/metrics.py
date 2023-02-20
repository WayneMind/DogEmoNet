import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def accuracy(output, target, topk=(1,)):
    """
    计算预测准确率的函数。该函数支持计算多个准确率，返回的结果为一个元组，元组中包含了每个指标的准确率
    :param output: 模型的输出。一般是模型的最终分类得分或概率
    :param target: 真实标签。一般是一个一维张量或数组，包含每个样本的真实标签
    :param topk: 一个元组，表示计算几个不同的top k准确率
    :return: 一个元组，包含了每个指标的准确率。例如，如果topk=(1, 5)，则返回(准确率@1, 准确率@5)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))

        return tuple(result)


def precision_recall(output, target, average='macro'):
    """
    计算精确率和召回率的函数
    :param output: 模型的输出。一般是模型的最终分类得分或概率
    :param target: 真实标签。一般是一个一维张量或数组，包含每个样本的真实标签
    :param average: 可以选择'binary', 'micro', 'macro'中的一种来计算精确率和召回率的平均值
    :return: 返回一个元组，包含了精确率和召回率
    """
    with torch.no_grad():
        # 对输出进行 softmax，计算每个类别的概率
        probs = F.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)

        # 计算精确率和召回率
        precision, recall, _, _ = metrics.precision_recall_fscore_support(target, predicted, average=average)

        return precision, recall


def f1_score(output, target, average='macro'):
    """计算F1分数"""
    # 将输出转换为预测的类别
    _, preds = torch.max(output, 1)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(target.cpu().numpy(), preds.cpu().numpy())

    # 计算每个类别的精确度和召回率
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall)

    if average == 'macro':
        # 对每个类别的F1分数取平均
        f1 = np.mean(f1)
    elif average == 'micro':
        # 将所有的真阳性、假阳性和假阴性相加后计算F1分数
        tp = np.sum(np.diag(conf_matrix))
        fp = np.sum(conf_matrix, axis=0) - np.diag(conf_matrix)
        fn = np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        raise ValueError("Unsupported average option: {}. "
                         "Valid options are 'macro' and 'micro'".format(average))

    return f1


def auc(output, target):
    """Computes the area under the receiver operating characteristic (ROC) curve.

    Args:
        output (torch.Tensor): Model output of shape (batch_size, num_classes).
        target (torch.Tensor): Target labels of shape (batch_size,).

    Returns:
        float: AUC score.

    """
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        score = roc_auc_score(target.cpu(), output.cpu()[:, 1])
    return score
