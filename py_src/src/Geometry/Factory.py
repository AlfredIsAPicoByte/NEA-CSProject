from abc import ABC, abstractmethod

from .Core import Shape

class ShapeFactory(ABC):
    """Abstract factory for creating shapes."""
    @abstractmethod
    def create(self, **kwargs) -> Shape:
        raise NotImplementedError