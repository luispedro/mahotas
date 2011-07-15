# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

from __future__ import division
import numpy as np
import _center_of_mass

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

