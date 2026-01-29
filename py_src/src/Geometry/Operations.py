import numpy as np

"""
Geometry Constructive Solid Geometry (CSG) Operations.
Provides functions to combine signed distance fields (SDFs) using CSG operations.
Also includes:
    - Smooth variants for blending shapes
    - Standard operations: union, addition, subtraction, intersection
    - XOR operation for exclusive combinations
"""

def op_union(d1, d2):
    return np.minimum(d1, d2)

def op_smooth_union(d1, d2, k: float):
    k *= 4.0
    h = np.maximum(k - abs(d1 - d2), 0.0)
    return np.minimum(d1, d2) - (h ** 2) * 0.25 / k

def op_addition(d1, d2):
    return np.maximum(d1, -d2)

def op_smooth_addition(d1, d2, k):
    return op_smooth_union(d1, -d2, k)

def op_subtract(d1, d2): # is not commutative for sdf's and depending on the order of the operand it will produce different results
    return np.maximum(-d1, d2)

def op_smooth_subtract(d1, d2, k):
    return -op_smooth_union(d1, -d2, k)

def op_intersect(d1, d2):
    return np.maximum(d1, d2)

def op_smooth_intersect(d1, d2, k):
    return -op_smooth_subtract(-d1, -d2, k)

def op_xor(d1, d2):
    return np.maximum(np.minimum(d1, d2), -np.maximum(d1, d2))