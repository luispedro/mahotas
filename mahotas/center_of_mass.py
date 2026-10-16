# Copyright (C) 2008-2024, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING file)

import numpy as np
from numpy.typing import NDArray

from ._typing import Array
from . import _center_of_mass


def center_of_mass(img: Array, labels: Array | None = None) -> NDArray[np.float64]:
    '''
    coords = center_of_mass(img, labels=None)

    Returns the center of mass of img.

    Each pixel is weighted by its value in `img` (so that, for a binary image,
    this is the centroid of the foreground pixels). Coordinates are in the
    same order as the axes of `img` (i.e., ``(row, column)`` for 2-D images).

    If `labels` is given, then it returns ``labels.max() + 1`` centers of
    mass, one for each region identified by `labels` (including region 0).

    Parameters
    ----------
    img : ndarray
    labels : ndarray, optional
        A labeled array (i.e., an array of integers of the same shape as
        ``img`` such that each "object" is identified by areas with different
        values).

    Returns
    -------
    coords : ndarray
        The exact shape of the output depends on whether the ``labels``
        argument was used. If ``labels is None``, then the return value is a
        1-ndarray of coordinates (``size = len(img.shape)``); otherwise, the
        return value is a 2-ndarray of coordinates (``shape = (labels.max()+1,
        len(img.shape))``).
    '''
    if labels is not None:
        if labels.dtype != np.int32 or \
            not labels.flags['C_CONTIGUOUS']:
            labels = np.ascontiguousarray(labels, np.int32)
        else:
            # This is necessary because labels might be of a type that equals
            # NPY_INT32, but is not NPY_INT32
            labels = labels.view(np.int32)
    cm = _center_of_mass.center_of_mass(img, labels)
    if labels is not None:
        return cm.reshape((-1, img.ndim))
    return cm
