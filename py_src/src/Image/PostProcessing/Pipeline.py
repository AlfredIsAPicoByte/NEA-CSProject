from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class PostProcessPass(ABC):
    """
    Abstract base class for any image effect.
    """
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Receives a Float32 RGB image (H, W, 3).
        Returns the modified image.
        """
        pass

class ImagePipeline:
    """
    Orchestrates the chain of effects.
    """
    def __init__(self):
        self.passes: List[PostProcessPass] = []

    def add_pass(self, effect: PostProcessPass):
        self.passes.append(effect)

    def execute(self, input_image: np.ndarray) -> np.ndarray:
        """
        Runs the image through all registered passes in order.
        """
        # Work on a copy to avoid modifying the original Film data by accident
        current_image = input_image.copy()
        
        for effect in self.passes:
            current_image = effect.apply(current_image)
            
        return current_image