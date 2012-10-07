'''\
=======
Mahotas
=======

A package for computer vision in Python.

Main Features
-------------

features
    Compute global and local features (several submodules, include SURF and Haralick features)
convolve
    Convolution and wavelets
morph
    Morphological features. Most are available at the mahotas level, include erode(), dilate()...
watershed
    Seeded watershed implementation
imread/imsave
    read/write image

'''
try:
    from .bbox import bbox, croptobbox
    from .center_of_mass import center_of_mass
    from .convolve import convolve, convolve1d, median_filter, rank_filter, template_match, gaussian_filter1d, gaussian_filter
    from .convolve import haar, ihaar, daubechies, idaubechies, wavelet_center, wavelet_decenter
    from .distance import distance
    from .edge import sobel
    from .euler import euler
    from .histogram import fullhistogram
    from .labeled import border, borders, bwperim, label, labeled_sum
    from .features.moments import moments
    from .morph import cerode, close, close_holes, get_structuring_elem, dilate, hitmiss, erode, cwatershed, majority_filter, open, regmin, regmax
    from .resize import imresize
    from .stretch import stretch, as_rgb
    from .thin import thin
    from .thresholding import otsu, rc
    from .io import imread, imsave

    from .tests import run as test

    from mahotas_version import __version__

    import features
    import morph
    import segmentation
except ImportError, e:
    import sys
    print >>sys.stderr, '''\
Could not import submodules (exact error was: %s).

There are many reasons for this error the most common one is that you have
either not built the packages or have built (using `python setup.py build`) or
installed them (using `python setup.py install`) and then proceeded to test
mahotas **without changing the current directory**.

Try installing and then changing to another directory before importing mahotas.
''' % e



__all__ = [
    'as_rgb',
    'bbox',
    'border',
    'borders',
    'bwperim',
    'center_of_mass',
    'close_holes',
    'convolve',
    'convolve1d',
    'croptobbox',
    'cwatershed',
    'daubechies',
    'idaubechies',
    'dilate',
    'distance',
    'erode',
    'euler',
    'fullhistogram',
    'gaussian_filter',
    'gaussian_filter1d',
    'get_structuring_elem',
    'haar',
    'ihaar',
    'hitmiss',
    'imresize',
    'label',
    'labeled_sum',
    'majority_filter',
    'median_filter',
    'moments',
    'morph',
    'otsu',
    'rank_filter',
    'rc',
    'sobel',
    'stretch',
    'template_match',
    'thin',
    'features',
    'morph',
    'segmentation',

    'imread',
    'imsave',

    'test',

    '__version__',
    ]

