import argparse
import os
import yaml

import random
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import CustomDataset
from utils.transforms import get_transforms
from utils.metrics import get_accuracy
from utils.logger import get_logger
from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.lr_scheduler import get_lr_scheduler
from models.ResNet import ResNet18
from models.DenseNet import DenseNet121
from models.EfficientNet import EfficientNetB0
from models.Inception import InceptionV3
from models.VGG import VGG16
from utils.visualization import plot_loss_accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(config, device):
    """模型训练函数"""
    # 解析配置文件
    model_name = config['model']
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    milestones = config['milestones']
    gamma = config['gamma']
    early_stopping = config['early_stopping']
    train_csv_path = config['train_csv_path']
    val_csv_path = config['val_csv_path']
    checkpoints_path = config['checkpoints_path']
    logs_path = config['logs_path']

    # 创建保存训练日志的Logger
    logger = get_logger(logs_path, model_name)

    # 创建训练集和验证集的数据集
    train_transforms = get_transforms('train')
    val_transforms = get_transforms('val')
    train_dataset = AudioDataset(train_csv_path, train_transforms)
    val_dataset = AudioDataset(val_csv_path, val_transforms)

    # 创建训练集和验证集的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 加载模型
    if model_name == 'ResNet18':
        model = ResNet18(num_classes)
    elif model_name == 'DenseNet121':
        model = DenseNet121(num_classes)
    elif model_name == 'EfficientNetB0':
        model = EfficientNetB0(num_classes)
    elif model_name == 'InceptionV3':
        model = InceptionV3(num_classes)
    elif model_name == 'VGG16':
        model = VGG16(num_classes)
    else:
        raise ValueError(f'Invalid model name: {model_name}')

    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 加载最近保存的检查点，如果有的话
    start_epoch = 0
    best_accuracy = 0
    if os.path.isfile(checkpoints_path):
        start_epoch, model, optimizer, best_accuracy = load_checkpoint(checkpoints_path, model, optimizer)

    # 设置学习率衰减策略
    lr_scheduler = get_lr_scheduler(optimizer, milestones, gamma)

    # 开始模型训练
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_train_samples = 0

        # 迭代训练集
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算训练集上的准确率
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            train_acc += correct
            num_train_samples += labels.size(0)

            # 累加损失
            train_loss += loss.item()

            # 每隔一定batch_size打印一次日志
            if (i + 1) % batch_size == 0:
                # 打印训练进度
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), train_loss / batch_size,
                              train_acc / num_train_samples * 100))
                # 记录训练日志
                logger.add_scalar('Train/Loss', train_loss / batch_size, epoch * len(train_loader) + i)
                logger.add_scalar('Train/Accuracy', train_acc / num_train_samples * 100, epoch * len(train_loader) + i)

                train_loss = 0.0
                train_acc = 0.0
                num_train_samples = 0

        # 每训练一个epoch，在验证集上测试模型
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            num_val_samples = 0

            for i, data in enumerate(val_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 计算验证集上的准确率
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                val_acc += correct
                num_val_samples += labels.size(0)

                # 累加损失
                val_loss += loss.item()

            # 记录验证日志
            logger.add_scalar('Val/Loss', val_loss / len(val_loader), epoch)
            logger.add_scalar('Val/Accuracy', val_acc / num_val_samples * 100, epoch)

            # 打印每个epoch的验证结果
            print('Epoch [{}/{}], Val Loss: {:.4f}, Val Acc: {:.2f}%'
                  .format(epoch + 1, num_epochs, val_loss / len(val_loader),
                          val_acc / num_val_samples * 100))

            # 更新模型保存
            checkpoint.save(model, optimizer, epoch, val_acc / num_val_samples * 100)

    # 训练结束
    print("Finished training!")


def load_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def main():
    # 加载配置文件
    config = load_config('experiments/experiment_DenseNet/config.yaml')

    # 从配置文件中获取模型名称
    model_name = config['model']['name']

    # 根据模型名称导入模型类
    if model_name == 'DenseNet':
        from models.DenseNet import DenseNet

        model = DenseNet(**config['model']['params'])
    elif model_name == 'EfficientNet':
        from models.EfficientNet import EfficientNet

        model = EfficientNet(**config['model']['params'])
    elif model_name == 'Inception':
        from models.Inception import Inception

        model = Inception(**config['model']['params'])
    elif model_name == 'ResNet':
        from models.ResNet import ResNet

        model = ResNet(**config['model']['params'])
    elif model_name == 'VGG':
        from models.VGG import VGG

        model = VGG(**config['model']['params'])

    # 从配置文件中获取数据集路径和其他超参数
    dataset_path = config['data']['path']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']

    # Set seed for reproducibility
    set_seed(42)
    # Load configuration
    config = load_config(args.config_file)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Train model
    train(config, device)


if __name__ == '__main__':
    main()
