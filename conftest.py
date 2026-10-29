"""
conftest.py
============
Root pytest configuration for the Optimized EDOS test suite.
Ensures the project root is on sys.path so all `src.*` imports resolve correctly.
"""

import sys
import os

# Add the project root directory to Python's module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
