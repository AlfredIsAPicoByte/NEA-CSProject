from math import gcd

class Ratio:
    """
    Represents a ratio (fraction) with a width and height.
    Supports basic arithmetic operations and comparisons.
    """
    def __init__(self, width: float, height: float):
        """
        Creates a ratio (fraction) with a width and height.
        """
        if width == 0:
            raise ValueError("Width (denominator) cannot be zero")
        
        if height == 0:
            raise ValueError("Height (numerator) cannot be zero")
            
        self.width = width
        self.height = height

    @classmethod
    def simplify(cls):
        """Simplify this Ratio in-place and return self (e.g., 1920/1080 -> 16/9)."""
        w_int = int(cls.width)
        h_int = int(cls.height)

        divisor = gcd(w_int, h_int)
        if divisor == 0:
            return cls

        # Use integer division to keep them integral
        cls.width = w_int // divisor
        cls.height = h_int // divisor
        return cls

    def __add__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        
        new_height = self.height * other.width + other.height * self.width
        new_width = self.width * other.width
        return Ratio(new_width, new_height)
    
    def __sub__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
            
        new_height = self.height * other.width - other.height * self.width
        new_width = self.width * other.width
        return Ratio(new_width, new_height)

    def __mul__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
            
        new_height = self.height * other.height
        new_width = self.width * other.width
        return Ratio(new_width, new_height)
    
    def __truediv__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        if other.height == 0:
            raise ValueError("Cannot divide by a Ratio with a height of zero")
            
        new_height = self.height * other.width
        new_width = self.width * other.height
        return Ratio(new_width, new_height)
    
    def __repr__(self):
        return f"Ratio({self.height}/{self.width})"
    
    def __float__(self):
        return self.height / self.width
    
    def __neg__(self):
        return Ratio(-self.width, -self.height)
    
    def __eq__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width == self.width * other.height
    
    def __lt__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width < self.width * other.height
        
    def __le__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width <= self.width * other.height
        
    def __gt__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width > self.width * other.height
        
    def __ge__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width >= self.width * other.height
        
    def __ne__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width != self.width * other.height
    
    @property
    def value(self) -> float:
        """Returns the float value of the ratio."""
        return float(self.width / self.height)
    