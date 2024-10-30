from pathlib import Path

import torch
import torchvision
from torchvision.transforms.v2 import functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: torch.nn.Module):
        self.paths: list[str] = []
        self.images: list[torch.Tensor | None] = []
        for img_path in Path(root).glob('*.png'):
            self.paths.append(img_path)
            self.images.append(None)
        self.transform = transform
        self.elastic_transform = torchvision.transforms.ElasticTransform()
        self.random_perspective = torchvision.transforms.RandomPerspective()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        if self.images[idx] is None:
            self.images[idx] = self.transform(torchvision.io.read_image(self.paths[idx]))
        image = self.images[idx]
        image = F.gaussian_noise_image(image)
        image = F.gaussian_blur_image(image, kernel_size=(3, 3), sigma=0.1)
        image = self.elastic_transform.forward(image)
        target = self.images[idx]
        return image, target
