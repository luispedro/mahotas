import numpy as np
from mahotas import freeimage
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
