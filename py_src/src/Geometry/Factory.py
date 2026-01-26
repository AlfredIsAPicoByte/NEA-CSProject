from abc import ABC, abstractmethod

from .SDF import SignedDistanceShape

class ShapeFactory(ABC):
    """
    Abstract Factory for creating Shape instances.
    """
    @abstractmethod
    def create_shape(self) -> SignedDistanceShape:
        pass