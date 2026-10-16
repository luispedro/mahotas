"""Static typing smoke tests; run with mypy --follow-imports=silent."""

from pathlib import Path
from typing import Any, assert_type

import numpy as np
from numpy.typing import NDArray

import mahotas as mh
from mahotas import labeled, morph
from mahotas.features import haralick, lbp, zernike_moments
from mahotas.interpolate import shift, zoom


def check_api(image: NDArray[np.uint8], flag: bool) -> None:
    assert_type(mh.dilate(image), NDArray[np.uint8])
    assert_type(mh.erode(image, Bc=4, out=image), NDArray[np.uint8])
    assert_type(mh.convolve(image, image), NDArray[np.uint8])
    assert_type(mh.convolve1d(image, [1, 2, 1], axis=0), NDArray[np.uint8])
    assert_type(mh.resize_to(image, (32, 32)), NDArray[np.uint8])
    assert_type(mh.thin(image), NDArray[np.uint8])
    assert_type(mh.locmax(image), NDArray[np.bool_])
    assert_type(mh.distance(image), NDArray[np.float64])
    assert_type(mh.fullhistogram(image), NDArray[np.uintc])
    assert_type(mh.otsu(image), int)
    assert_type(mh.citation(print_out=False), str)
    assert_type(mh.imread(Path('image.png')), NDArray[Any])
    mh.imsave(Path('image.png'), image)

    labels, count = mh.label(image)
    assert_type(labels, NDArray[np.int32])
    assert_type(count, int)
    assert_type(morph.cwatershed(image, labels), NDArray[np.int64])
    assert_type(
        mh.cwatershed(image, labels, return_lines=True),
        tuple[NDArray[np.int64], NDArray[np.bool_]],
    )
    assert_type(
        mh.cwatershed(image, labels, None, True),
        tuple[NDArray[np.int64], NDArray[np.bool_]],
    )
    assert_type(
        mh.cwatershed(image, labels, return_lines=flag),
        NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.bool_]],
    )
    assert_type(mh.border(labels, 1, 2), NDArray[np.bool_])
    assert_type(mh.border(labels, 1, 2, always_return=False), NDArray[np.bool_] | None)
    assert_type(labeled.bbox(labels), NDArray[np.intp])
    assert_type(labeled.bbox(labels, as_slice=True), list[tuple[slice, ...]])
    assert_type(mh.bbox(image), NDArray[np.intp])

    assert_type(mh.dog(image), NDArray[np.bool_])
    assert_type(mh.dog(image, just_filter=True), NDArray[np.float64])
    assert_type(mh.sobel(image, just_filter=True), NDArray[np.float64])
    assert_type(mh.sobel(image), NDArray[np.bool_] | NDArray[np.float64])
    assert_type(zoom(image, 2), NDArray[np.float64])
    assert_type(zoom(image, 1, out=image), NDArray[np.uint8])
    assert_type(shift(image, [1, 1]), NDArray[np.float64])
    assert_type(haralick(image), NDArray[np.float64])
    assert_type(lbp(image, radius=1, points=8), NDArray[np.float64])
    assert_type(zernike_moments(image, radius=8), NDArray[np.float64])
