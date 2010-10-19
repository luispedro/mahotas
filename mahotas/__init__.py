'''\
=======
Mahotas
=======

A set of functions for basic image processing and machine vision.
'''
from .histogram import fullhistogram
from .thresholding import otsu, rc
from .stretch import stretch
import morph
from .morph import close_holes, get_structuring_elem, dilate, erode, cwatershed, majority_filter
from .thin import thin
from .bwperim import bwperim
from .center_of_mass import center_of_mass
from .bbox import bbox, croptobbox
from .edge import sobel
from .euler import euler
from .moments import moments
from .distance import distance
from .resize import imresize
from mahotas_version import __version__

import segmentation
import features

try:
    from .freeimage import imread, imsave
except OSError, e:
    def imread(*args, **kwargs):
        raise ImportError('mahotas.imread dependends on freeimage. Could not find it. Error was: %s' % e)
    def imsave(*args, **kwargs):
        raise ImportError('mahotas.imread dependends on freeimage. Could not find it. Error was: %s' % e)

