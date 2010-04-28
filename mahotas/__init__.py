'''\
=======
Mahotas
=======

A set of functions for basic image processing and machine vision.
'''
from .histogram import fullhistogram
from .thresholding import otsu
from .stretch import stretch
import morph
from .morph import *
from mahotas_version import __version__

__doc__=morph.__doc__
