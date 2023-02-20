import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, metrics=['loss', 'accuracy']):
    """Plots the training history of a model.

    Args:
        history (dict): A dictionary of training history.
        metrics (list): A list of metrics to plot.

    """
    for metric in metrics:
        plt.plot(history[metric], label=metric)
    plt.legend()
    plt.show()


def plot_confusion_matrix(cm, class_names, normalize=False):
    """Plots a confusion matrix.

    Args:
        cm (numpy.ndarray): A confusion matrix.
        class_names (list): A list of class names.
        normalize (bool): If True, normalize the confusion matrix.

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(title='Confusion matrix', xlabel='Predicted label', ylabel='True label')


def plot_roc_curve(fpr, tpr, roc_auc):
    """Plots the ROC curve.

    Args:
        fpr (numpy.ndarray): False positive rates.
        tpr (numpy.ndarray): True positive rates.
        roc_auc (float): Area under the ROC curve (AUC).

    """
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()


def save_figure(fig, filename):
    """Saves a figure to a file in the experiment directory.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        filename (str): Name of the file to save the figure to.

    Returns:
        None

    """
    experiment_dir = os.path.join(os.getcwd(), '../experiments')
    experiment_name = os.path.basename(os.getcwd())
    results_dir = os.path.join(experiment_dir, experiment_name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, filename)
    fig.savefig(file_path)


# # 例子: 使用 save_figure 函数保存图片到指定路径
# fig = plt.figure()
# plt.plot([0, 1], [0, 1])
# plt.title('Example figure')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# save_figure(fig, 'example.png')
