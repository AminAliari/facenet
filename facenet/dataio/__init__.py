"""Package containing dataset and data loaders.

Author
 * Mohammadamin Aliari
"""


import os

__all__ = []
for filename in os.listdir(os.path.dirname(__file__)):
    filename = os.path.basename(filename)
    if filename.endswith(".py") and not filename.startswith("__"):
        __all__.append(filename[:-3])

from . import *  # noqa
