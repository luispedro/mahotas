from __future__ import division
import numpy as np

def imresize(img, nsize, order=3):
    '''
    img' = imresize(img, nsize)
    
    Resizes img

    Parameters
    ----------
    img : ndarray
    nsize : float or tuple(float) or tuple(integers)
        Size of return. Meaning depends on the type
            float: img'.shape[i] = nsize * img.shape[i]
            tuple of float: img'.shape[i] = nsize[i] * img.shape[i]
            tuple of int: img'.shape[i] = nsize[i]
    order : integer, optional
        Spline order to use (default: 3)

    Returns
    -------
    img' : ndarray

    See Also
    --------
    ``scipy.ndimage.zoom`` and ``scipy.misc.pilutil.imresize``
    '''
    from .interpolate import zoom
    if type(nsize) == tuple:
        if type(nsize[0]) == int:
            nsize = np.array(nsize, dtype=float)
            nsize /= img.shape
    return zoom(img, nsize, order=order)
