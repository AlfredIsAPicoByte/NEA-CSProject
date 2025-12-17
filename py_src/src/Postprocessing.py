import math
import numpy as np
from typing import List, Tuple

from Luminance import Color

class PostProcessor:
    @staticmethod
    def aces_tone_map(color: Color) -> Color:
        """
        ACES (Academy Color Encoding System) Filmic Tone Mapping.
        Great contrast, nice saturation, and handles bright lights gracefully.
        Adapted from Narkowicz 2015.
        """
        # ACES constants
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        
        # Helper to map a single channel
        def map_channel(v: float) -> float:
            # Formula: (x * (a * x + b)) / (x * (c * x + d) + e)
            return (v * (a * v + b)) / (v * (c * v + d) + e)

        return Color(
            map_channel(color.red),
            map_channel(color.green),
            map_channel(color.blue)
        )

    @staticmethod
    def reinhard_tone_map(color: Color) -> Color:
        """
        Classic Reinhard Tone Mapping.
        Simple logic: color / (color + 1).
        Preserves details but tends to look grey/flat in highlights.
        """
        def map_channel(v: float) -> float:
            return v / (v + 1.0)
            
        return Color(
            map_channel(color.red),
            map_channel(color.green),
            map_channel(color.blue)
        )

    @staticmethod
    def gamma_correct(color: Color, gamma: float = 2.2) -> Color:
        """
        Converts Linear Space -> sRGB Gamma Space.
        Monitor standard is usually Gamma 2.2.
        """
        inv_gamma = 1.0 / gamma
        return Color(
            max(0, color.red) ** inv_gamma,
            max(0, color.green) ** inv_gamma,
            max(0, color.blue) ** inv_gamma
        )