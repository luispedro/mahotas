# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Copyright (C) 2013  Luis Pedro Coelho
# 
# License: MIT (see COPYING file)

from matplotlib import pyplot as plt
import numpy as np

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
    img = plt.imread(filename)
    if as_grey and len(img.shape) == 3:
        # these are the values that wikipedia says are typical
        transform = np.array([0.30, 0.59, 0.11])
        return np.dot(img, transform)
    return img

imsave = plt.imsave
