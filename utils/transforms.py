import random
import torchvision.transforms as transforms


class RandomRotation:
    """随机旋转图像"""

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        # 随机生成旋转角度
        angle = random.uniform(-self.degrees, self.degrees)
        # 旋转图像
        return transforms.functional.rotate(img, angle)


class RandomHorizontalFlip:
    """随机水平翻转图像"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        # 以概率p随机水平翻转图像
        if random.random() < self.p:
            return transforms.functional.hflip(img)
        else:
            return img


"""
这里定义了两个图像变换类，分别为RandomRotation和RandomHorizontalFlip。

RandomRotation用于对图像进行随机旋转变换。它接收一个degrees参数，表示旋转的最大角度，然后在变换时随机生成一个-degrees到degrees之间的角度，对图像进行旋转变换。

RandomHorizontalFlip用于对图像进行随机水平翻转变换。它接收一个p参数，表示水平翻转的概率，默认为0.5。在变换时，以p的概率对图像进行水平翻转，以增加数据的多样性。
"""