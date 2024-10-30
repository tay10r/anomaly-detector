from io import BytesIO

from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
import zmq

from src.task import Task
from src.report_writer import ReportWriter

class Evaluator(Task):
    def __init__(self,
                 module: nn.Module,
                 loader: DataLoader,
                 device,
                 report_writer: ReportWriter):
        self.module = module
        self.loader = loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.report_writer = report_writer
        self.epoch = 0

    def step(self):
        loss = 0.0
        self.module.eval()
        self.report_writer.begin_report(epoch=self.epoch)
        for sample in self.loader:
            image, target = sample
            image = image.to(self.device)
            target = target.to(self.device)
            result = self.module(image)
            loss += self.criterion(result, target).item()
            self.report_writer.report(image, result, target)
        avg_loss = loss / len(self.loader)
        self.report_writer.end_report(test_loss=avg_loss)
        logger.info(f'Test loss: {avg_loss}')
        self.epoch += 1
