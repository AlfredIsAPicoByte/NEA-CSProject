from abc import ABC, abstractmethod
import numpy as np

from .SDF import *

class ShapeFactory(ABC):
    """
    Abstract Factory for creating Shape instances.
    """
    @abstractmethod
    def create(self, *args, **kwargs) -> SignedDistanceShape:
        pass