import signal
import argparse

from loguru import logger
from torch.utils.data import DataLoader
from torch import nn, device
from torch import float32
from torch import cuda
import torch

import torchvision.transforms.v2 as transforms

import zmq

from augmentation import Transform

from src.loop import Loop
from src.dataset import Dataset
from src.tasks.optimizer import Optimizer
from src.tasks.evaluator import Evaluator
from src.nn.registry import create_module
from src.report_writer import ZmqReportWriter

global keep_going

def on_signal(signum: int, frame):
    logger.info(f'Signal {signum} caught, exiting.')
    global keep_going
    keep_going = False

def export_module(module: nn.Module):
    module = module.cpu().eval()
    input_ = torch.randn(1, 3, 120, 120, requires_grad=True)
    torch.onnx.export(module,
                      input_,
                      'model.onnx',
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={ 'input': { 0: 'batch_size' }, 'output': { 0: 'batch_size' }})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, required=False)
    parser.add_argument('--learning-rate', type=float, default=0.0001, required=False)
    parser.add_argument('--model', type=str, default='v1_x1', required=False)
    args = parser.parse_args()

    if cuda.is_available():
        dev = device('cuda')
        pin_memory = True
        logger.info('Using CUDA device.')
    else:
        dev = device('cpu')
        pin_memory = False
        logger.info('Using CPU device.')

    global keep_going
    keep_going = True
    signal.signal(signal.SIGINT, on_signal)

    transform = Transform()
    transform.set_infill_rect(x=40, y=40, w=40, h=40)
    transform.set_noise_range(0, 0)

    train_data = Dataset(root='data/train', transform=transform)
    test_data = Dataset(root='data/test', transform=transform)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)#, num_workers=2)#, pin_memory=pin_memory, pin_memory_device=dev)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)#, num_workers=2)#, pin_memory=pin_memory, pin_memory_device=dev)

    module = create_module(args.model)
    module = module.to(dev)
    logger.info(f'Training data has {len(train_data)} samples.')
    logger.info(f'Test data has {len(test_data)} samples.')
    logger.info('Starting loop.')
    zmq_context = zmq.Context()
    reporter = ZmqReportWriter(transform, zmq_context)
    loop = Loop()
    loop.add_task(Optimizer(train_loader, module, learning_rate=args.learning_rate, dev=dev))
    loop.add_task(Evaluator(module, test_loader, device=dev, report_writer=reporter))
    while keep_going:
        loop.step()
        logger.info('Step complete.')
    logger.info('Loop terminated.')
    logger.info('Exporting module.')
    export_module(module)
    logger.info('Exiting')

main()