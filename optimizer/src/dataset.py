from pathlib import Path

import torch
from torch import unsqueeze
import torchvision
from torchvision.transforms.v2 import functional as F

# Note: This module is defined in 'csrc/'. You have to install it to your venv with `pip install csrc/augmentation`
from augmentation import Transform, open_image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: torch.nn.Module):
        self.paths: list[str] = []
        for entry in Path(root).glob('*.jpg'):
            self.paths.append(entry)
        self.transform = Transform()
        self.transform.set_noise_levels(min_value=0, max_value=64)
        self.transform.set_infill_sizes(x_min=16, y_min=16, x_max=64, y_max=64)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        """
        if self.images[idx] is None:
            self.images[idx] = self.transform(torchvision.io.read_image(self.paths[idx], torchvision.io.ImageReadMode.RGB))
        image = self.images[idx]
        image = F.gaussian_noise_image(image)
        image = F.adjust_hue_image(image, hue_factor=0.05)
        target = self.images[idx]
        return image, target
        """
        img = open_image(str(self.paths[idx]))
        target = img
        img = self.transform(img)
        img = F.to_dtype(torch.from_numpy(img), scale=True)
        target = F.to_dtype(torch.from_numpy(target), scale=True)
        return img, target

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: torch.nn.Module):
        self.paths = [
            str(Path(root) / 'input.jpg')
        ]
        self.transform = Transform()
        self.transform.set_noise_levels(min_value=0, max_value=64)
        self.transform.set_infill_sizes(x_min=16, y_min=16, x_max=64, y_max=64)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        img = open_image(str(self.paths[idx]))
        target = img
        img = self.transform(img)
        img = F.to_dtype(torch.from_numpy(img), scale=True)
        target = F.to_dtype(torch.from_numpy(target), scale=True)
        return img, target