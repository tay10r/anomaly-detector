from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from io import BytesIO

import torch
from torch import Tensor, concat, nn
from torch.nn.functional import mse_loss
from torchvision.utils import save_image, make_grid

import zmq

class ReportWriter(ABC):
    """
    Used for reporting the optimization process.
    """
    def __init__(self):
        pass

    @abstractmethod
    def begin_report(self, epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def end_report(self, test_loss: float):
        raise NotImplementedError()

    @abstractmethod
    def report(self, image: Tensor, result: Tensor, target: Tensor):
        raise NotImplementedError()

@dataclass
class _ZmqReport:
    epoch: int
    test_loss: float
    per_sample_mse: list[float]

class ZmqReportWriter(ReportWriter):
    """
    Reports the results of each evaluation cycle through a set of ZMQ sockets.
    """
    def __init__(self,
                 zmq_context: zmq.Context,
                 image_pub_address: str = 'tcp://*:6021',
                 metrics_pub_address: str = 'tcp://*:6022'):
        self.image_socket = zmq.Socket(zmq_context, zmq.PUB)
        self.image_socket.bind(image_pub_address)
        self.metrics_socket = zmq.Socket(zmq_context, zmq.PUB)
        self.metrics_socket.bind(metrics_pub_address)
        self.current_report = _ZmqReport(epoch=0, test_loss=0, per_sample_mse=[])
        self.first_image_sent = False

    def begin_report(self, epoch: int):
        self.current_report = _ZmqReport(epoch=epoch, test_loss=0, per_sample_mse=[])
        self.first_image_sent = False

    def end_report(self, test_loss: float):
        self.current_report.test_loss = test_loss
        self.metrics_socket.send_json(asdict(self.current_report))

    def report(self, image: Tensor, result: Tensor, target: Tensor):

        if not self.first_image_sent:
            delta = (result - target)**2
            example = concat((image, result, delta), dim=0)
            g = make_grid(example)
            buffer = BytesIO()
            save_image(g, buffer, 'png')
            self.image_socket.send(buffer.getvalue())
            self.first_image_sent = True

        mse = []
        for i in range(image.shape[0]):
            mse.append(mse_loss(result[i], target[i]).item())
        self.current_report.per_sample_mse += mse
