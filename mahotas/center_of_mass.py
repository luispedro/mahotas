# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np
from . import _center_of_mass

def center_of_mass(img, labels=None):
    '''
    coords = center_of_mass(img, labels=None)
    x0,x1,... = center_of_mass(img, labels=None)

    Returns the center of mass of img.

    If `labels` is given, then it returns `L` centers of mass, one for each
    region identified by `labels` (including region 0).

    Parameters
    ----------
    img : ndarray
    labels : ndarray
        A labeled array

    Returns
    -------
    coords : ndarray
        if ``not labels``, a 1-ndarray of coordinates (size = len(img.shape)),
        if ``labels``, a 2-ndarray of coordinates (shape = (labels.max()+1) xlen(img.shape))
    '''
    if labels is not None:
        if labels.dtype != np.int32 or \
            not labels.flags['C_CONTIGUOUS']:
            labels = np.ascontiguousarray(labels, np.int32)
        else:
            # This is necessary because it might be of a type that equals
            # NPY_INT32, but is not NPY_INT32
            labels = labels.view(np.int32)
    cm = _center_of_mass.center_of_mass(img, labels)
    if labels is not None:
        return cm.reshape((-1, img.ndim))
    return cm

