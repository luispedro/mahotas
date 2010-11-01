# -*- coding: utf-8 -*-
# Copyright (C) 2008-2009  Murphy Lab, Carnegie Mellon University
#
# Written by Robert Webb and Luis Pedro Coelho <lpc@cmu.edu>
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
# For additional information visit http://murphylab.web.cmu.edu or
# send email to murphy@cmu.edu

import numpy as np
from scipy import ndimage
from histogram import fullhistogram

def _roll_left(v, points):
    return (v >> 1) | ( (1 << (points-1)) * (v & 1) )

def _precompute_mapping(points):
    res = np.zeros(2**points, np.uint32)
    from ._lbp import map
    map(res, points)
    return res


def lbp(image, radius, points):
    '''
    features = lbp(image, radius, points)

    Compute Linear Binary Patterns

    Parameters
    ----------
    image : ndarray
        input image (2-D numpy ndarray)
    radius : number (integer or floating point
        radius (in pixels)
    points : integer
        nr of points to consider

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
    from scipy.ndimage.interpolation import map_coordinates
    import mahotas._lbp
    from mahotas.histogram import fullhistogram
    Y,X = np.indices(image.shape)
    angles = np.linspace(0, 2*np.pi, points+1)[:-1]
    coordinates = np.empty((2, points, image.size), float)
    for i,(dy,dx) in enumerate(zip(radius * np.sin(angles), radius * np.cos(angles))):
        coordinates[0][i] = Y.ravel()
        coordinates[1][i] = X.ravel()
        coordinates[0][i] += dy
        coordinates[1][i] += dx
    data = map_coordinates(image, coordinates.reshape((2,-1)), order=1).reshape((image.size, -1))
    codes = (data.T > image.ravel()).sum(0)
    mahotas._lbp.map(codes.astype(np.uint32), points)
    final = fullhistogram(codes.astype(np.uint32))

    codes = np.arange(2**points, dtype=np.uint32)
    iters = codes.copy()
    mahotas._lbp.map(codes.astype(np.uint32), points)
    pivots = (codes == iters)
    npivots = np.sum(pivots)
    compressed = final[pivots[:len(final)]]
    compressed = np.concatenate((compressed, [0]*(npivots - len(compressed))))
    return compressed
