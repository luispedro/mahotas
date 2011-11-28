# -*- coding: utf-8 -*-
# Copyright (C) 2008-2009 Robert Webb and Luis Pedro Coelho <luis@luispedro.org>
# Copyright (C) 2011 Luis Pedro Coelho <luis@luispedro.org>
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
#

import numpy as np
from histogram import fullhistogram

__all__ = [
    'lbp',
    ]
def lbp(image, radius, points, ignore_zeros=False):
    '''
    features = lbp(image, radius, points, ignore_zeros=False)

    Compute Linear Binary Patterns

    Parameters
    ----------
    image : ndarray
        input image (2-D numpy ndarray)
    radius : number (integer or floating point)
        radius (in pixels)
    points : integer
        nr of points to consider
    ignore_zeros : boolean, optional
        whether to ignore zeros (default: False)

    Returns
    -------
    features : 1-D numpy ndarray
        histogram of features


    Reference
    ---------
    Gray Scale and Rotation Invariant Texture Classification with Local Binary Patterns
        Ojala, T. Pietikainen, M. Maenpaa, T. LECTURE NOTES IN COMPUTER SCIENCE (Springer)
        2000, ISSU 1842, pages 404-420
    '''
    from .interpolate import shift
    import mahotas._lbp
    from mahotas.histogram import fullhistogram
    if ignore_zeros:
        Y,X = np.nonzero(image)
        def select(im):
            return im[Y,X].ravel()
        pixels = image[Y,X].ravel()
    else:
        select = np.ravel
        pixels = image.ravel()
    image = image.astype(np.float64)
    angles = np.linspace(0, 2*np.pi, points+1)[:-1]
    data = []
    for dy,dx in zip(np.sin(angles), np.cos(angles)):
        data.append(
            select(shift(image, [radius*dy,radius*dx], order=1)))
    data = np.array(data)
    codes = (data > pixels).astype(np.int32)
    codes *= (2**np.arange(points)[:,np.newaxis])
    codes = codes.sum(0)
    codes = mahotas._lbp.map(codes.astype(np.uint32), points)
    final = fullhistogram(codes.astype(np.uint32))

    codes = np.arange(2**points, dtype=np.uint32)
    iters = codes.copy()
    codes = mahotas._lbp.map(codes.astype(np.uint32), points)
    pivots = (codes == iters)
    npivots = np.sum(pivots)
    compressed = final[pivots[:len(final)]]
    compressed = np.concatenate((compressed, [0]*(npivots - len(compressed))))
    return compressed
