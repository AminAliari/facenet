""" Package containing real face detector "facenet".

Author
 * Mohammadamin Aliari
"""

import os

from . import nnet  # noqa
from . import dataio  # noqa
from .core import Config  # noqa

__all__ = [
    "Config",
]

with open(os.path.join(os.path.dirname(__file__), "version.txt")) as f:
    version = f.read().strip()

__version__ = version
