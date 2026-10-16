import mahotas as mh
from mahotas import imresize
import numpy as np
import pytest
def test_imresize():
    img = np.repeat(np.arange(100), 10).reshape((100,10))
    assert imresize(img, (1024,104)).shape == (1024,104)
    assert imresize(img, (10.,10.)).shape == (1000,100)
    assert imresize(img, .2,).shape == (20,2)
    assert imresize(img, (10.,2.)).shape == (1000,20)


def test_resize_to():
    lena  = mh.demos.load('lena')
    im = mh.resize.resize_rgb_to(lena, [256,256])
    assert im.shape == (256,256,3)
    im = im.max(2)
    im  = mh.resize.resize_to(im, [512,256])
    assert im.shape == (512,256)

def test_resize_edge_cases():
    for start,target in [
            ((128,128), (64,64)),
            ((127,128), (64,64)),
            ((125,128), (64,64)),
            ((120,128), (64,64)),
            ((117,128), (64,64)),
            ((115,128), (61,64)),
            ((116,129), (59,65)),
            ]:
        assert mh.resize_to(np.zeros(start), target).shape == target
    start = np.zeros((65,31))
    for t0 in range(7,29):
        assert mh.resize_to(start, (t0, 17)).shape == (t0, 17)


@pytest.mark.parametrize('shape, target', [
    ((49, 55), (64, 224)),
    ((47, 71), (96, 224)),
    ((97, 94), (224, 384)),
])
@pytest.mark.parametrize('container', [tuple, list])
@pytest.mark.parametrize('dtype', [np.uint8, np.float32])
def test_imresize_integer_size_is_exact(shape, target, container, dtype):
    image = (np.arange(shape[0])[:, None] + np.arange(shape[1])).astype(dtype)
    result = imresize(image, container(target), order=1)
    assert result.shape == target
    assert result.dtype == np.float64
    other = imresize(np.zeros((32, 32), dtype=dtype), container(target), order=1)
    assert np.stack([result, other]).shape == (2,) + target


def test_imresize_numpy_integer_size():
    image = np.zeros((49, 55))
    target = tuple(np.array([64, 224]))
    assert isinstance(target[0], np.integer)
    assert imresize(image, target, order=1).shape == (64, 224)
