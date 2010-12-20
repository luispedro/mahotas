'''\
=======
Mahotas
=======

A set of functions for basic image processing and machine vision.
'''
from .bbox import bbox, croptobbox
from .bwperim import bwperim
from .center_of_mass import center_of_mass
from .convolve import convolve
from .distance import distance
from .edge import sobel
from .euler import euler
from .histogram import fullhistogram
from .moments import moments
from .morph import close_holes, get_structuring_elem, dilate, erode, cwatershed, majority_filter
from .resize import imresize
from .stretch import stretch
from .thin import thin
from .thresholding import otsu, rc

from mahotas_version import __version__

import features
import morph
import segmentation

__all__ = [
    'bbox',
    'bwperim',
    'center_of_mass',
    'close_holes',
    'convolve',
    'croptobbox',
    'cwatershed',
    'dilate',
    'distance',
    'erode',
    'euler',
    'fullhistogram',
    'get_structuring_elem',
    'imresize',
    'majority_filter',
    'moments',
    'morph',
    'otsu',
    'rc',
    'sobel',
    'stretch',
    'thin',

    'features',
    'morph',
    'segmentation',

    '__version__',
    ]

try:
    from .freeimage import imread, imsave
    __all__ += [
        'imread',
        'imsave',
        ]
except OSError, e:
    def imread(*args, **kwargs):
        raise ImportError('mahotas.imread dependends on freeimage. Could not find it. Error was: %s' % e)
    def imsave(*args, **kwargs):
        raise ImportError('mahotas.imread dependends on freeimage. Could not find it. Error was: %s' % e)

