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
from .freeimage import imread, imsave
from mahotas_version import __version__

import segmentation
import features
