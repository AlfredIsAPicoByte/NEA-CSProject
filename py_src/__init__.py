"""NEA-CSProject py_src package initializer.

This file makes the package import-safe and documents package exports.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

sys.path.insert(0, current_dir)