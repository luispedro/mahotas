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
from .morph import close_holes, get_structuring_elem, dilate, erode, cwatershed
from mahotas_version import __version__

