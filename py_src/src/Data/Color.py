import numpy as np
import numbers
from dataclasses import dataclass, replace
from typing import Callable, List, Tuple, Union
import bisect

from src.Utilities.Common import clamp, lerp

import numpy as np
import numbers
from dataclasses import dataclass, replace
from typing import Callable, List, Tuple, Union
import bisect
import math

# Assuming clamp and lerp are available from your common utilities
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

    # Backwards compatible attribute names used in some tests
    @property
    def red(self) -> float:
        return self.r

    @property
    def green(self) -> float:
        return self.g

    @property
    def blue(self) -> float:
        return self.b

    def to_np_ndarray(self) -> np.ndarray:
        """Returns np.array([r,g,b,a]) compatible with legacy callers."""
        return np.array([self.r, self.g, self.b, self.a])

    # =========================================================================
    # STATIC CONVERTERS (Color Spaces & Formats)
    # =========================================================================

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

    @staticmethod
    def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
        """Converts HSL (0.0-1.0) to RGB (0.0-1.0)."""
        def hue_to_rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p

        if s == 0:
            return l, l, l

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
        return r, g, b

    @staticmethod
    def rgb_to_hsl(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Converts RGB (0.0-1.0) to HSL (0.0-1.0)."""
        mx = max(r, g, b)
        mn = min(r, g, b)
        h, s, l = 0.0, 0.0, (mx + mn) / 2

        if mx == mn:
            h = 0.0
            s = 0.0 # Achromatic
        else:
            d = mx - mn
            s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
            
            if mx == r:
                h = (g - b) / d + (6 if g < b else 0)
            elif mx == g:
                h = (b - r) / d + 2
            elif mx == b:
                h = (r - g) / d + 4
            h /= 6
            
        return h, s, l

    @staticmethod
    def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> Tuple[float, float, float]:
        """Converts CMYK (0.0-1.0) to RGB (0.0-1.0)."""
        r = (1.0 - c) * (1.0 - k)
        g = (1.0 - m) * (1.0 - k)
        b = (1.0 - y) * (1.0 - k)
        return r, g, b

    @staticmethod
    def rgb_to_cmyk(r: float, g: float, b: float) -> Tuple[float, float, float, float]:
        """Converts RGB (0.0-1.0) to CMYK (0.0-1.0)."""
        if r == 0 and g == 0 and b == 0:
            return 0.0, 0.0, 0.0, 1.0
        
        k = 1.0 - max(r, g, b)
        c = (1.0 - r - k) / (1.0 - k)
        m = (1.0 - g - k) / (1.0 - k)
        y = (1.0 - b - k) / (1.0 - k)
        return c, m, y, k

    @staticmethod
    def rgb_to_grayscale(r: float, g: float, b: float) -> float:
        """Standard luminance conversion (Rec. 709)."""
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @staticmethod
    def hex_to_rgb(hex_str: str) -> Tuple[float, float, float, float]:
        """
        Parses hex strings: #RGB, #RGBA, #RRGGBB, #RRGGBBAA.
        Returns (r, g, b, a) as floats 0.0-1.0.
        """
        hex_str = hex_str.lstrip('#')
        length = len(hex_str)
        
        r, g, b, a = 0.0, 0.0, 0.0, 1.0
        
        if length == 3: # RGB
            r = int(hex_str[0]*2, 16) / 255.0
            g = int(hex_str[1]*2, 16) / 255.0
            b = int(hex_str[2]*2, 16) / 255.0
        elif length == 4: # RGBA
            r = int(hex_str[0]*2, 16) / 255.0
            g = int(hex_str[1]*2, 16) / 255.0
            b = int(hex_str[2]*2, 16) / 255.0
            a = int(hex_str[3]*2, 16) / 255.0
        elif length == 6: # RRGGBB
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
        elif length == 8: # RRGGBBAA
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            a = int(hex_str[6:8], 16) / 255.0
        else:
            raise ValueError(f"Invalid hex string format: {hex_str}")
            
        return r, g, b, a

    @staticmethod
    def rgb_to_hex(r: float, g: float, b: float, a: float = 1.0, include_alpha: bool = False) -> str:
        """Returns hex string #RRGGBB or #RRGGBBAA."""
        ri = clamp(int(r * 255), 0, 255)
        gi = clamp(int(g * 255), 0, 255)
        bi = clamp(int(b * 255), 0, 255)
        
        if include_alpha:
            ai = clamp(int(a * 255), 0, 255)
            return f"#{ri:02X}{gi:02X}{bi:02X}{ai:02X}"
        return f"#{ri:02X}{gi:02X}{bi:02X}"

    @staticmethod
    def kelvin_to_rgb(temp_k: float) -> Tuple[float, float, float]:
        """
        Approximates RGB from color temperature (Kelvin).
        Valid range roughly 1000K to 40000K.
        """
        temp = clamp(temp_k, 1000.0, 40000.0) / 100.0
        
        # Red
        if temp <= 66:
            r = 255.0
        else:
            r = temp - 60
            r = 329.698727446 * (r ** -0.1332047592)
            
        # Green
        if temp <= 66:
            g = temp
            g = 99.4708025861 * math.log(g) - 161.1195681661
        else:
            g = temp - 60
            g = 288.1221695283 * (g ** -0.0755148492)
            
        # Blue
        if temp >= 66:
            b = 255.0
        elif temp <= 19:
            b = 0.0
        else:
            b = temp - 10
            b = 138.5177312231 * math.log(b) - 305.0447927307

        return (
            clamp(r / 255.0, 0.0, 1.0),
            clamp(g / 255.0, 0.0, 1.0),
            clamp(b / 255.0, 0.0, 1.0)
        )

    @staticmethod
    def wavelength_to_rgb(wavelength: float) -> Tuple[float, float, float]:
        """
        Converts light wavelength (nm) to RGB.
        Valid range ~380nm to ~780nm.
        """
        wl = float(wavelength)
        gamma = 0.8
        
        r, g, b = 0.0, 0.0, 0.0
        factor = 0.0

        if 380 <= wl < 440:
            r = -(wl - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif 440 <= wl < 490:
            r = 0.0
            g = (wl - 440) / (490 - 440)
            b = 1.0
        elif 490 <= wl < 510:
            r = 0.0
            g = 1.0
            b = -(wl - 510) / (510 - 490)
        elif 510 <= wl < 580:
            r = (wl - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif 580 <= wl < 645:
            r = 1.0
            g = -(wl - 645) / (645 - 580)
            b = 0.0
        elif 645 <= wl <= 780:
            r = 1.0
            g = 0.0
            b = 0.0
        else:
            r = 0.0
            g = 0.0
            b = 0.0

        # Let the intensity fall off near the vision limits
        if 380 <= wl < 420:
            factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
        elif 420 <= wl < 700:
            factor = 1.0
        elif 700 <= wl <= 780:
            factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
        else:
            factor = 0.0

        def adjust(c, factor):
            if c == 0.0: return 0.0
            return (c * factor) ** gamma

        return adjust(r, factor), adjust(g, factor), adjust(b, factor)

    # =========================================================================
    # CONSTRUCTORS / FACTORY METHODS
    # =========================================================================

    @classmethod
    def from_hex(cls, hex_str: str):
        r, g, b, a = cls.hex_to_rgb(hex_str)
        return cls(r, g, b, a)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, a: float = 1.0):
        # Accept either hue in degrees (0..360) or fraction (0..1)
        hh = float(h)
        ss = float(s)
        vv = float(v)
        if hh > 1.0:
            hh = hh / 360.0
        if ss > 1.0:
            ss = ss / 100.0
        if vv > 1.0:
            vv = vv / 100.0

        r, g, b = cls.hsv_to_rgb(hh, ss, vv)
        return cls(r, g, b, a)

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, a: float = 1.0):
        # Accept hue in degrees or fraction, saturation/light as percent or fraction
        hh = float(h)
        ss = float(s)
        ll = float(l)
        if hh > 1.0:
            hh = hh / 360.0
        if ss > 1.0:
            ss = ss / 100.0
        if ll > 1.0:
            ll = ll / 100.0

        r, g, b = cls.hsl_to_rgb(hh, ss, ll)
        return cls(r, g, b, a)

    @classmethod
    def from_cmyk(cls, c: float, m: float, y: float, k: float, a: float = 1.0):
        r, g, b = cls.cmyk_to_rgb(c, m, y, k)
        return cls(r, g, b, a)

    @classmethod
    def from_kelvin(cls, k: float, a: float = 1.0):
        r, g, b = cls.kelvin_to_rgb(k)
        return cls(r, g, b, a)
    
    @classmethod
    def from_wavelength(cls, nm: float, a: float = 1.0):
        r, g, b = cls.wavelength_to_rgb(nm)
        return cls(r, g, b, a)

    @classmethod
    def from_int_rgb(cls, r: int, g: int, b: int, a: int = 255):
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    @classmethod
    def from_np(cls, arr: np.ndarray):
        """Creates color from numpy array (size 3 or 4) or returns Color unchanged."""
        # If already a Color, return as-is
        if isinstance(arr, Color):
            return arr

        a = np.asarray(arr)
        if a.ndim == 0:
            raise ValueError("Input cannot be a scalar for Color.from_np")
        if a.shape[0] == 3:
            return cls(float(a[0]), float(a[1]), float(a[2]), 1.0)
        elif a.shape[0] >= 4:
            return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))
        raise ValueError("Numpy array must have at least 3 elements")

    # =========================================================================
    # INSTANCE CONVERSION METHODS
    # =========================================================================
    
    def to_hsv(self) -> Tuple[float, float, float]:
        return self.rgb_to_hsv(self.r, self.g, self.b)

    def to_hsl(self) -> Tuple[float, float, float]:
        return self.rgb_to_hsl(self.r, self.g, self.b)
    
    def to_cmyk(self) -> Tuple[float, float, float, float]:
        return self.rgb_to_cmyk(self.r, self.g, self.b)

    def to_grayscale(self) -> 'Color':
        lum = self.rgb_to_grayscale(self.r, self.g, self.b)
        return Color(lum, lum, lum, self.a)
    
    def to_hex(self, include_alpha: bool = False) -> str:
        return self.rgb_to_hex(self.r, self.g, self.b, self.a, include_alpha)

    def to_int_rgb(self) -> Tuple[int, int, int]:
        return (
            int(clamp(int(self.r * 255), 0, 255)),
            int(clamp(int(self.g * 255), 0, 255)),
            int(clamp(int(self.b * 255), 0, 255))
        )
    
    def to_int_rgba(self) -> Tuple[int, int, int, int]:
        return (
            int(clamp(int(self.r * 255), 0, 255)),
            int(clamp(int(self.g * 255), 0, 255)),
            int(clamp(int(self.b * 255), 0, 255)),
            int(clamp(int(self.a * 255), 0, 255))
        )

    def to_np_array(self, include_alpha: bool = False) -> np.ndarray:
        """Return color as numpy array. Defaults to RGB (no alpha)."""
        if include_alpha:
            return np.array([self.r, self.g, self.b, self.a], dtype=np.float32)
        return np.array([self.r, self.g, self.b], dtype=np.float32)

    # --- Operator Overloads (Kept from original) ---
    def __add__(self, other: Union['Color', float, int, np.number]):
        if isinstance(other, numbers.Number):
            return Color(self.r + other, self.g + other, self.b + other, self.a)
        if isinstance(other, Color):
            return Color(self.r + other.r, self.g + other.g, self.b + other.b, self.a)
        return NotImplemented

    def __radd__(self, other: Union['Color', float, int, np.number]):
        return self.__add__(other)

    def __sub__(self, other: Union['Color', float, int, np.number]):
        if isinstance(other, numbers.Number):
            return Color(self.r - other, self.g - other, self.b - other, self.a)
        if isinstance(other, Color):
            return Color(self.r - other.r, self.g - other.g, self.b - other.b, self.a)
        return NotImplemented

    def __rsub__(self, other: Union['Color', float, int, np.number]):
        if isinstance(other, numbers.Number):
            return Color(other - self.r, other - self.g, other - self.b, self.a)
        return NotImplemented

    def __mul__(self, scale: Union['Color', float, int, np.number]):
        if isinstance(scale, numbers.Number):
            return Color(self.r * scale, self.g * scale, self.b * scale, self.a)
        if isinstance(scale, Color):
            return Color(self.r * scale.r, self.g * scale.g, self.b * scale.b, self.a * scale.a)
        return NotImplemented

    def __rmul__(self, scale: Union['Color', float, int, np.number]):
        return self.__mul__(scale)

    def __truediv__(self, scale: Union[float, int, np.number]):
        if isinstance(scale, numbers.Number) and scale != 0:
            recip = 1.0 / scale
            return Color(self.r * recip, self.g * recip, self.b * recip, self.a)
        return NotImplemented
    
    def __rtruediv__(self, other: Union[float, int, np.number]):
        if isinstance(other, numbers.Number):
             return Color(other / (self.r + 1e-8), other / (self.g + 1e-8), other / (self.b + 1e-8), self.a)
        return NotImplemented
    
    def __getitem__(self, index):
        if index == 0 or index == "r" or index == "red":
            return self.r
        elif index == 1 or index == "g" or index == "green":
            return self.g 
        elif index == 2 or index == "b" or index == "blue":
            return self.b
        elif index == 3 or index == "a" or index == "alpha":
            return self.a
        raise IndexError(f"Invalid index for Color: {index}")

    def __repr__(self):
        return f"Color(r={self.r:.2f}, g={self.g:.2f}, b={self.b:.2f}, a={self.a:.2f})"

@dataclass(slots=True)
class ColorGradient:
    colors: List[Color]
    positions: np.ndarray
    interpolation: Callable[[float], float] = lambda x: x  # Linear by default

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
        factor = self.interpolation(local_t)

        # 5. Vectorized Linear Interpolation (Lerp)
        # Calculates R, G, B, A simultaneously
        return lerp(c0, c1, factor)