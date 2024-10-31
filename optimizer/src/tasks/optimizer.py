from torch.utils.data import DataLoader
from torch.nn import Module, MSELoss
from torch.optim import Adam

from loguru import logger

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
        loss_sum = 0.0
        for sample in self.loader:
            image, target = sample
            image = image.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            result = self.module(image)
            loss = self.criterion(result, target)
            loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()
        loss_avg = loss_sum / len(self.loader)
        logger.info(f'Training loss: {loss_avg}')