'''
Pillow-based image input/output

This backend is used by ``mahotas.io`` when the ``imread`` package is not
installed, but Pillow (or PIL) is.
'''
import numpy as np
import mahotas as mh
from PIL import Image

def imread(filename, as_grey=False):
    '''Read an image into a ndarray from a file.

    This function depends on PIL (or Pillow) being installed.

    Parameters
    ----------
    filename : str
        filename
    as_grey : boolean, optional
        Whether to convert to grey scale image (default: no)

    Returns
    -------
    array : ndarray
        Image data. Colour images are returned as ``(height, width, channels)``
        arrays. If `as_grey` is true, colour images are converted with
        ``mahotas.colors.rgb2grey`` (which returns a floating point array).
    '''
    im = Image.open(filename)
    array = np.array(im)
    if as_grey and array.ndim != 2:
        array = mh.colors.rgb2grey(array)
    return array

def imsave(filename, array):
    '''
    Writes `array` into file `filename`

    This function depends on PIL (or Pillow) being installed.

    Parameters
    ----------
    filename : str
        path on file system
    array : ndarray-like
        Image data. It must be of a type/shape which Pillow's
        ``Image.fromarray`` can handle. The file format is inferred from
        the extension of `filename`.
    '''
    im = Image.fromarray(array)
    im.save(filename)

