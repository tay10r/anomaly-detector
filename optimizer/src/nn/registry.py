from typing import Callable

from torch.nn import Module

from src.nn.networks import Network

global registry

registry: dict[str, Callable] = {
    'v1_x1': lambda: Network(n=1),
    'v1_x2': lambda: Network(n=2),
    'v1_x4': lambda: Network(n=4)
}

def register_module(name: str, factory: Callable):
    global registry
    registry[name] = factory

def create_module(name: str) -> Module:
    global registry
    return registry[name]()
