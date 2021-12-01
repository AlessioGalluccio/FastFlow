"""
Framework for Easily Invertible Architectures.
Module to construct invertible networks with pytorch, based on a graph
structure of operations.
"""
from . import framework
from . import modules

__all__ = ["framework", "modules"]
