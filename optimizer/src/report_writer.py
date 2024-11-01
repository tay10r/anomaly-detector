from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from io import BytesIO

from torch import Tensor, from_numpy, FloatTensor, ByteTensor
from torch.nn.functional import mse_loss
from torchvision.utils import save_image, make_grid

import zmq

from augmentation import Transform

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
                 transform: Transform,
                 zmq_context: zmq.Context,
                 image_pub_address: str = 'tcp://*:6021',
                 metrics_pub_address: str = 'tcp://*:6022'):
        self.image_socket = zmq.Socket(zmq_context, zmq.PUB)
        self.image_socket.bind(image_pub_address)
        self.metrics_socket = zmq.Socket(zmq_context, zmq.PUB)
        self.metrics_socket.bind(metrics_pub_address)
        self.current_report = _ZmqReport(epoch=0, test_loss=0, per_sample_mse=[])
        self.transform = transform
        self.results: list[Tensor] = []

    def begin_report(self, epoch: int):
        self.current_report = _ZmqReport(epoch=epoch, test_loss=0, per_sample_mse=[])
        self.results.clear()

    def end_report(self, test_loss: float):
        self.current_report.test_loss = test_loss
        self.metrics_socket.send_json(asdict(self.current_report))

        buffer = BytesIO()
        g = make_grid(self.results, nrow=14)
        save_image(g, buffer, 'png')
        self.image_socket.send(buffer.getvalue())

    def report(self, image: Tensor, result: Tensor, target: Tensor):
        for i in range(result.shape[0]):
            self.results.append(result[i])
        mse = []
        for i in range(image.shape[0]):
            mse.append(mse_loss(result[i], target[i]).item())
        self.current_report.per_sample_mse += mse
