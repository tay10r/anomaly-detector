from pathlib import Path

import torch
from torch import unsqueeze
from torchvision.transforms.v2 import functional as F

# Note: This module is defined in 'csrc/'. You have to install it to your venv with `pip install csrc/augmentation`
from augmentation import Transform, open_image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: Transform):
        self.paths: list[str] = []
        for entry in Path(root).glob('*.png'):
            self.paths.append(entry)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        img = open_image(str(self.paths[idx]))
        img, target = self.transform(img)
        img = F.to_dtype(torch.from_numpy(img), scale=True)
        target = F.to_dtype(torch.from_numpy(target), scale=True)
        return img, target