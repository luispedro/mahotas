import numpy as np
from mahotas import freeimage
from mahotas import imread,imsave
import os
from os import path
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

def test_rgba():
    rgba = path.join(
                path.dirname(__file__),
                'data',
                'rgba.png')
    rgba = imread(rgba)
    assert np.all(np.diff(rgba[:,:,3].mean(1)) < 0 ) # the image contains an alpha gradient


@with_setup(teardown=_remove_image)
def test_save_load_rgba():
    img = np.arange(256).reshape((8,8,4)).astype(np.uint8)
    freeimage.imsave(_testimgname, img)
    img_ = freeimage.imread(_testimgname)
    assert img.shape == img_.shape
    assert np.all(img == img_)

def test_fromblob():
    img = np.arange(100, dtype=np.uint8).reshape((10,10))
    s = freeimage.imsavetoblob(img, 't.png')
    assert np.all(freeimage.imreadfromblob(s) == img)

    s = freeimage.imsavetoblob(img, 't.bmp')
    assert np.all(freeimage.imreadfromblob(s) == img)


