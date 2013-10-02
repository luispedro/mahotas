# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Copyright (C) 2013  Luis Pedro Coelho
# 
# License: MIT (see COPYING file)

import numpy as np

# Importing matplotlib checks that it is importable without triggering any
# initialization (unlike importing pyplot)
import matplotlib

def imread(filename, as_grey=False):
    """
    img = imread(filename, as_grey=False)

    Reads an image from file `filename`

    Parameters
    ----------
      filename : file name
      as_grey : Whether to convert to grey scale image (default: no)

    Returns
    -------
      img : ndarray
    """
    from matplotlib import pyplot as plt
    img = plt.imread(filename)
    if as_grey and len(img.shape) == 3:
        # these are the values that wikipedia says are typical
        transform = np.array([0.30, 0.59, 0.11])
        return np.dot(img, transform)
    return img

def imsave(filename, array):
    '''
    imsave(filename, array)

    Writes `array` into file `filename`

    Parameters
    ----------
    filename : str
        path on file system
    array : ndarray-like
    '''
    from matplotlib import pyplot as plt
    import numpy as np
    if len(array.shape) == 2:
        import warnings
        warnings.warn('mahotas.imsave: The `matplotlib` backend does not support saving greyscale images natively.\n'
                    'Emulating by saving in RGB format (with all channels set to same value).\n'
                    'If this is a problem, please use another IO backend\n'
                    '\n'
                    'See http://mahotas.readthedocs.org/en/latest/io.html \n'
                    )
        array = np.dstack([array, array, array])
    plt.imsave(filename, array)

