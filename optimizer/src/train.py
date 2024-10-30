import signal
import argparse

from loguru import logger
from torch.utils.data import DataLoader
from torch import nn, device
from torch import float32
from torch import cuda
import torchvision.transforms.v2 as transforms

import zmq

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

def get_transforms() -> nn.Module:
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(float32, scale=True)
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, required=False)
    parser.add_argument('--learning-rate', type=float, default=0.01, required=False)
    parser.add_argument('--model', type=str, default='v2', required=False)
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
    train_data = Dataset(root='data/train', transform=get_transforms())
    test_data = Dataset(root='data/test', transform=get_transforms())
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)#, num_workers=2)#, pin_memory=pin_memory, pin_memory_device=dev)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)#, num_workers=2)#, pin_memory=pin_memory, pin_memory_device=dev)
    module = create_module(args.model)
    module = module.to(dev)
    logger.info(f'Training data has {len(train_data)} samples.')
    logger.info(f'Test data has {len(test_data)} samples.')
    logger.info('Starting loop.')
    zmq_context = zmq.Context()
    reporter = ZmqReportWriter(zmq_context)
    loop = Loop()
    loop.add_task(Optimizer(train_loader, module, learning_rate=args.learning_rate, dev=dev))
    loop.add_task(Evaluator(module, test_loader, device=dev, report_writer=reporter))
    while keep_going:
        loop.step()
        logger.info('Step complete.')
    logger.info('Done')

main()