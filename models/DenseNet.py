import torch.nn as nn
import torchvision.models as models


class DenseNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DenseNet, self).__init__()

        # Load the pretrained DenseNet model
        self.model = models.densenet121(pretrained=pretrained)

        # Replace the last fully connected layer with a new one
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


"""
DenseNet是一种卷积神经网络架构，主要由DenseBlock和TransitionBlock两个部分组成。
DenseBlock可以看作是一种网络块的结构，其中每一层的输入包含前面所有层的特征图。
而TransitionBlock则是一个类似于ResNet中的过渡块，用于调整通道数和空间分辨率。
"""
