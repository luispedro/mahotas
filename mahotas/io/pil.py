import numpy as np
from PIL import Image

def imread(filename):
    '''Read an image into a ndarray from a file.

    This function depends on PIL (or Pillow) being installed.

    Parameters
    ----------
    filename : str
        filename
    '''
    array = Image.open(filename)
    return np.array(array)

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

