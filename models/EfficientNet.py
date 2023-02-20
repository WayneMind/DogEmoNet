import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=3, efficientnet_type='b0'):
        """
        构造函数
        :param num_classes: 分类数
        :param efficientnet_type: EfficientNet 的类型，支持 'b0' 到 'b7'
        """
        super(EfficientNetModel, self).__init__()

        # 加载预训练模型
        self.efficientnet = EfficientNet.from_pretrained(f'efficientnet-{efficientnet_type}')

        # 获取 EfficientNet 最后一层的输出特征数
        num_ftrs = self.efficientnet._fc.in_features

        # 将 EfficientNet 最后一层改为 num_classes 输出
        self.efficientnet._fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量
        :return: 输出张量
        """
        x = self.efficientnet(x)
        return x



"""
这里使用了第三方库 efficientnet_pytorch，需要先安装才能正常运行。
EfficientNet 是由 Google 提出的一种轻量级卷积神经网络结构。
它的设计采用了一种新颖的复合缩放方法，可以显著提高神经网络在准确率和计算复杂度之间的平衡性。
"""
