import numpy as np
from mahotas import freeimage
from mahotas import imread,imsave
import os
from nose.tools import with_setup

_testimgname = '/tmp/mahotas_test.png'
def _remove_image():
    try:
        os.unlink(_testimgname)
    except OSError:
        pass

@with_setup(teardown=_remove_image)
def test_freeimage():
    img = np.arange(256).reshape((16,16)).astype(np.uint8)

    freeimage.imsave(_testimgname, img)
    img_ = freeimage.imread(_testimgname)
    assert img.shape == img_.shape
    assert np.all(img == img_)


@with_setup(teardown=_remove_image)
def test_as_grey():
    colour = np.arange(16*16*3).reshape((16,16,3))
    imsave(_testimgname, colour.astype(np.uint8))
    c2 = imread(_testimgname, as_grey=True)
    assert len(c2.shape) == 2
    assert c2.shape == colour.shape[:-1]
