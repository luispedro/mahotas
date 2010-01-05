'''\
==========
VISION
==========

A set of functions for basic image processing and machine vision.
'''
from .histogram import fullhistogram
from .thresholding import otsu
from .pit import stretch
import morph
from .morph import *
__doc__=morph.__doc__
