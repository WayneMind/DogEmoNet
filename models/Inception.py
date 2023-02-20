import torch.nn as nn
import torchvision.models as models


class Inception(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Inception, self).__init__()

        self.model = models.inception_v3(pretrained=pretrained)

        # Replace the last fully connected layer of Inception with a new one
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


"""
Inception类继承自nn.Module，重写了__init__方法和forward方法，以创建自己的Inception模型。
在__init__方法中，通过调用torchvision.models.inception_v3()函数加载Inception V3模型的预训练权重，
设置pretrained参数为True表示使用预训练权重。

通过替换Inception模型的最后一层全连接层，将模型的输出维度设置为num_classes，这里默认为2，表示二分类问题。
forward方法实现了模型的前向传播，将输入数据x通过Inception模型，最终输出预测结果。
"""
