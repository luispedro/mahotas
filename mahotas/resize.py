# Copyright (C) 2010-2012, Luis Pedro Coelho <luis@luispedro.org>
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

__all__ = [
    'imresize',
    ]

def imresize(img, nsize, order=3):
    '''
    img' = imresize(img, nsize)

    Resizes img

    Parameters
    ----------
    img : ndarray
    nsize : float or tuple(float) or tuple(integers)
        Size of return. Meaning depends on the type
            float: img'.shape[i] = nsize * img.shape[i]
            tuple of float: img'.shape[i] = nsize[i] * img.shape[i]
            tuple of int: img'.shape[i] = nsize[i]
    order : integer, optional
        Spline order to use (default: 3)

    Returns
    -------
    img' : ndarray

    See Also
    --------
    scipy.ndimage.zoom : Similar function
    scipy.misc.pilutil.imresize : Similar function
    '''
    from .interpolate import zoom
    if type(nsize) == tuple:
        if type(nsize[0]) == int:
            nsize = np.array(nsize, dtype=float)
            nsize /= img.shape
    return zoom(img, nsize, order=order)
