from torchvision import transforms
from PIL import Image

IMAGENETNORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
CIFAR10NORM = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]


class transform_evl:
    def __init__(self, image_size, train, normalize=CIFAR10NORM):
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)