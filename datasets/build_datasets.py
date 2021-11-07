import torch
import torch.utils.data


class BuildDataset(torch.utils.data.Dataset):
    """
    Build the dataset from video.
    """
    def __init__(self, root=None, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.size = 1000

    def __getitem__(self, idx):
        if idx < self.size:
            return [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))], [0,0,0]
        else:
            raise Exception

    def __len__(self):
        return self.size
