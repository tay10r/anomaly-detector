from torch.utils.data import DataLoader
from torch.nn import Module, MSELoss
from torch.optim import Adam

from src.task import Task

class Optimizer(Task):
    def __init__(self, loader: DataLoader, module: Module, learning_rate: float, dev):
        self.loader = loader
        self.module = module
        self.optimizer = Adam(self.module.parameters(), lr=learning_rate)
        self.criterion = MSELoss()
        self.device = dev

    def step(self):
        self.module.train()
        for sample in self.loader:
            image, target = sample
            image = image.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            result = self.module(image)
            loss = self.criterion(result, target)
            loss.backward()
            self.optimizer.step()