from torchvision import transforms
from .GaussianBlur import GaussianBlur

IMAGENETNORM = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
CIFAR10NORM = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]


class transform_denseCL:
    def __init__(self, image_size, normalize=CIFAR10NORM):
        image_size = 224 if image_size is None else image_size
        p_blur = 0.5 if image_size > 32 else 0
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=p_blur),
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
