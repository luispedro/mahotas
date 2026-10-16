# Copyright (C) 2008-2026, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

from typing import Literal, overload

from . import _bbox
import numpy as np
from numpy.typing import NDArray

from ._typing import Array, ScalarT

@overload
def bbox(img: Array, border: int | None = None, as_slice: Literal[False] = False) -> NDArray[np.intp]: ...

@overload
def bbox(
    img: Array, border: int | None = None, as_slice: bool = False,
) -> NDArray[np.intp] | tuple[slice, ...]: ...

def bbox(
    img: Array, border: int | None = None, as_slice: bool = False,
) -> NDArray[np.intp] | tuple[slice, ...]:
    """
    min1,max1,min2,max2 = bbox(img, border={0}, as_slice={False})

    Calculate the bounding box of image img.

    Works for arrays of any dimensionality, not just 2-D.

    Parameters
    ----------
    img : ndarray
        Any integer image type
    border : int, optional
        Pad the bounding box by this many pixels on each side, clipped to the
        image shape (default: 0, i.e. tight bounding box).
    as_slice : bool, optional
        If True, return a tuple of slices instead of integer coordinates.

    Returns
    -------
    coords : ndarray of intp, shape ``(2 * img.ndim,)``
        Interleaved ``(min_0, max_0, min_1, max_1, ...)`` such that
        ``img[min_0:max_0, min_1:max_1, ...]`` contains all non-zero pixels.
        Returned when ``as_slice`` is false (the default). For 2-D images this
        reduces to ``(min1, max1, min2, max2)``. If `img` has no non-zero
        pixels, all values are 0.
    s : tuple of slice
        A slice representation of the bounding box (one slice per axis).
        Returned when ``as_slice`` is true.

    Examples
    --------
    >>> import numpy as np
    >>> import mahotas as mh
    >>> f = np.zeros((6, 6), bool)
    >>> f[1:3, 2:5] = True
    >>> mh.bbox(f)
    array([1, 3, 2, 5])
    >>> f[mh.bbox(f, as_slice=True)].shape
    (2, 3)
    """
    if not img.shape:
        return np.array([], dtype=np.intp)
    r = _bbox.bbox(img)
    if border:
        r = r.reshape((-1, 2))
        np.maximum(r.T[0] - border, 0, out=r.T[0])
        r.T[1] += border
        r = r.ravel()
    if as_slice:
        r = tuple([slice(s,e) for s,e in r.reshape((-1,2))])
    return r

def croptobbox(img: NDArray[ScalarT], border: int | None = None) -> NDArray[ScalarT]:
    """
    nimg = croptobbox(img, border=0)

    Returns a version of img cropped to the image's bounding box

    Parameters
    ----------
    img : ndarray
        Integer image array
    border : int, optional
        Number of pixels to add around the bounding box (default: 0)

    Returns
    -------
    nimg : ndarray
        A subimage of img.

    Notes
    -----
    The ``border`` is applied to the bounding box and then clipped to the
    input shape, so the result is always a sub-image of ``img``. As a
    consequence, if the input already has non-zero pixels touching its own
    margin, those pixels will still be on the margin of the returned image
    -- the border cannot extend past the input.
    """
    sl = bbox(img, border=border, as_slice=True)
    return img[sl]
