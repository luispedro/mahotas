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
    '''
    im = Image.fromarray(array)
    im.save(filename)

