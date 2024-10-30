from typing import Callable

from torch.nn import Module

from src.nn.v1 import AutoEncoder
from src.nn.v2 import AutoEncoderV2

global registry

registry: dict[str, Callable] = {
    'v1': AutoEncoder,
    'v2': AutoEncoderV2
}

def register_module(name: str, factory: Callable):
    global registry
    registry[name] = factory

def create_module(name: str) -> Module:
    global registry
    return registry[name]()
