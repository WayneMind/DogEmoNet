import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()

        # 加载预训练的 VGG-16 模型
        self.vgg16 = models.vgg16(pretrained=True)

        # 替换最后的全连接层，使其输出 num_classes 个类别的概率分布
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.vgg16.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg16.classifier(x)
        return x


"""
注意：如果你要使用 VGG-16 进行训练，你需要下载 Imagenet 数据集的预训练权重。
在 PyTorch 中，你可以通过设置 pretrained=True 来自动下载预训练权重。如果你的计算机无法联网，你需要手动下载预训练权重并将其加载到模型中。
"""
