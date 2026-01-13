import numpy as np
import numbers
from dataclasses import dataclass, replace
from typing import Callable, List, Tuple, Union
import bisect

from src.Utilities.Common import clamp, lerp

@dataclass(slots=True)
class Color:
    """
    A data class representing a color with red, green, and blue components.
    Internal representation uses floats from 0.0 to 1.0.
    """
    r: float
    g: float
    b: float
    a: float = 1.0

    def clamp(self):
        """Clamps internal RGBA values to be between 0.0 and 1.0."""
        self.r = clamp(self.r)
        self.g = clamp(self.g)
        self.b = clamp(self.b)
        self.a = clamp(self.a)

    # --- Static Methods ---
    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Converts HSV (0.0-1.0) to RGB (0.0-1.0)."""
        if s == 0.0:
            return v, v, v
        
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        if i == 0: return v, t, p
        if i == 1: return q, v, p
        if i == 2: return p, v, t
        if i == 3: return p, q, v
        if i == 4: return t, p, v
        if i == 5: return v, p, q
        return 0.0, 0.0, 0.0

    @staticmethod
    def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Converts RGB (0.0-1.0) to HSV (0.0-1.0)."""
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        
        h = 0.0
        if mx == mn:
            h = 0.0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
            
        return (h / 360.0, (0 if mx == 0 else df / mx), mx)

    # --- Constructors ---
    @classmethod
    def from_hex(cls, hex_str: str):
        hex_str = hex_str.lstrip('#')
        if len(hex_str) != 6:
            raise ValueError(f"Invalid hex string: {hex_str}")
        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        return cls(r / 255.0, g / 255.0, b / 255.0)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, a: float = 1.0):
        r, g, b = cls.hsv_to_rgb(h, s, v)
        return cls(r, g, b, a)

    @classmethod
    def from_int_rgb(cls, r: int, g: int, b: int, a: int = 255):
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    # --- Output / Conversions ---
    def __add__(self, other: Union['Color', float, int, np.number]):
        """Handles: Color + Color, Color + float"""
        if isinstance(other, numbers.Number):
            return Color(self.r + other, self.g + other, self.b + other, self.a)
        if isinstance(other, Color):
            return Color(self.r + other.r, self.g + other.g, self.b + other.b, self.a)
        return NotImplemented

    def __radd__(self, other: Union['Color', float, int, np.number]):
        """Handles: float + Color"""
        return self.__add__(other)

    def __sub__(self, other: Union['Color', float, int, np.number]):
        """Handles: Color - Color, Color - float"""
        if isinstance(other, numbers.Number):
            return Color(self.r - other, self.g - other, self.b - other, self.a)
        if isinstance(other, Color):
            return Color(self.r - other.r, self.g - other.g, self.b - other.b, self.a)
        return NotImplemented

    def __rsub__(self, other: Union['Color', float, int, np.number]):
        """
        Handles: float - Color
        Example: 1.0 - Color(0.2, 0.2, 0.2) = Color(0.8, 0.8, 0.8)
        """
        if isinstance(other, numbers.Number):
            # precise order: number - color_component
            return Color(other - self.r, other - self.g, other - self.b, self.a)
        return NotImplemented

    def __mul__(self, scale: Union['Color', float, int, np.number]):
        """Handles: Color * Color, Color * float"""
        if isinstance(scale, numbers.Number):
            return Color(self.r * scale, self.g * scale, self.b * scale, self.a)
        if isinstance(scale, Color):
            return Color(self.r * scale.r, self.g * scale.g, self.b * scale.b, self.a * scale.a)
        return NotImplemented

    def __rmul__(self, scale: Union['Color', float, int, np.number]):
        """Handles: float * Color"""
        return self.__mul__(scale)

    def __truediv__(self, scale: Union[float, int, np.number]):
        """Handles: Color / float"""
        if isinstance(scale, numbers.Number) and scale != 0:
            recip = 1.0 / scale
            return Color(self.r * recip, self.g * recip, self.b * recip, self.a)
        return NotImplemented
    
    def __rtruediv__(self, other: Union[float, int, np.number]):
        """
        Handles: float / Color 
        Note: This is mathematically ambiguous for vectors, but usually implies 
        component-wise division: (x/r, x/g, x/b).
        """
        if isinstance(other, numbers.Number):
             return Color(other / (self.r + 1e-8), other / (self.g + 1e-8), other / (self.b + 1e-8), self.a)
        return NotImplemented
    
    def __getitem__(self, index):
        if index == 0 or index == "r" or index == "red":
            return self.r
        elif index == 1 or index == "g" or index == "green":
            return self.g  # Fixed
        elif index == 2 or index == "b" or index == "blue":
            return self.b  # Fixed
        elif index == 3 or index == "a" or index == "alpha":
            return self.a  # Fixed
        raise IndexError(f"Invalid index for Color: {index}")

    def __repr__(self):
        return f"Color(r={self.r:.2f}, g={self.g:.2f}, b={self.b:.2f}, a={self.a:.2f})"

@dataclass(slots=True)
class ColorGradient:
    colors: List[Color]
    positions: np.ndarray

    def __post_init__(self):
        """
        Args:
            positions: List of floats between 0.0 and 1.0 (must be sorted).
            colors: List of numpy arrays (e.g. [R, G, B, A]).
        """
        if len(self.positions) != len(self.colors):
            raise ValueError("Positions and colors must have the same length.")
        
        # Ensure sorted data for binary search logic
        sorted_pairs = sorted(zip(self.positions, self.colors), key=lambda x: x[0])
        self.positions = np.array([p for p, c in sorted_pairs], dtype=float)
        self.colors = [c for p, c in sorted_pairs]

    def get_color(
        self, 
        t: float, 
        interpolation_function: Callable[[float], float] = lambda x: x
    ) -> Color:
        """
        Get interpolated color at position t in [0.0, 1.0].
        Optimized using NumPy vectorization and bisect for speed.
        """
        # 1. Clamp t
        t = max(0.0, min(1.0, t))

        # 2. Fast Path: Boundaries
        if t <= self.positions[0]:
            return self.colors[0]
        if t >= self.positions[-1]:
            return self.colors[-1]

        # 3. Find the segment using binary search (faster than loop for many stops)
        # bisect_right returns the insertion point to maintain order. 
        # For t, it gives us the index of the first position > t.
        idx = bisect.bisect_right(self.positions, t)
        
        # The segment is between idx-1 and idx
        t0, t1 = self.positions[idx-1], self.positions[idx]
        c0, c1 = self.colors[idx-1], self.colors[idx]

        # 4. Calculate factor
        denom = t1 - t0
        if denom < 1e-8: # Avoid division by zero
            return c1
            
        local_t = (t - t0) / denom
        factor = interpolation_function(local_t)

        # 5. Vectorized Linear Interpolation (Lerp)
        # Calculates R, G, B, A simultaneously
        return lerp(c0, c1, factor)