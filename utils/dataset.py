import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    """自定义的数据集类，继承自torch.utils.data.Dataset"""

    def __init__(self, image_paths, targets, resize=None):
        """
        Args:
            image_paths (list): 图像路径列表
            targets (list): 目标标签列表
            resize (tuple): 图像调整大小的元组，格式为(width, height)
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        # 数据增强
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        从数据集中按索引获取一个样本。

        Args:
            index (int): 要获取的样本的索引

        Returns:
            sample (dict): 包含图像和标签的字典
        """
        # 从路径中加载图像
        image = Image.open(self.image_paths[index])
        # 如果resize参数不为空，则调整图像大小
        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)
        # 将图像转换为张量并进行归一化
        image = self.transform(image)
        # 获取标签
        target = self.targets[index]
        # 返回一个包含图像和标签的字典
        return {
            'image': image,
            'target': torch.tensor(target, dtype=torch.long)
        }

"""
这个文件定义了一个 CustomDataset 类，它是一个自定义的 PyTorch 数据集类。它继承自 PyTorch 中的 torch.utils.data.Dataset 类，这个类是所有自定义数据集类的基类。

在类的初始化方法中，传入两个参数 image_paths 和 targets，这是我们用于创建数据集的数据。resize 参数是一个元组，它指定图像将被调整为的大小。如果为 None，则不执行任何调整。

__len__ 方法返回数据集中的样本数量。

__getitem__ 方法从数据集中按索引获取一个样本。它打开路径中的图像，并将其转换为 PyTorch 张量。如果指定了 resize 参数，则调整图像大小。最后，它返回一个字典，其中包含图像张量和标签张量。
"""