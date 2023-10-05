"""
Helper library for machine learning combat for TDT4173 - Machine Learning 2023
"""

import os

module_dir = os.path.dirname(os.path.abspath(__file__))
py_files = [file[:-3] for file in os.listdir(module_dir) if file.endswith(".py") and file != '__init__.py']

__all__ = py_files

from . import data