import numpy as np
from mahotas.io import error_imread, error_imsave
from nose.tools import raises
from os import path
import mahotas as mh

filename = path.join(
            path.dirname(__file__),
            'data',
            'rgba.png')

@raises(ImportError)
def test_error_imread():
    error_imread(filename)

@raises(ImportError)
def test_error_imsave():
    error_imsave('/tmp/test_mahotas.png', np.arange(16, dtype=np.uint8).reshape((4,4)))


def test_as_grey():
    filename = path.join(
            path.dirname(__file__),
            '..',
            'demos',
            'data',
            'luispedro.jpg')
    im = mh.imread(filename, as_grey=1)
    assert im.ndim == 2
