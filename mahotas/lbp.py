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

from scipy import weave
from scipy.weave import converters

def _roll_left(v, points):
    return (v >> 1) | ( (1 << (points-1)) * (v & 1) )

def _precompute_mapping(points):
    res = np.zeros(2**points, np.int32)
    res -= 1
    for i in xrange(2**points):
        if res[i] == -1:
            bestval = i
            cur = i
            for j in xrange(points):
                cur = _roll_left(cur, points)
                if cur < bestval:
                    bestval = cur
            cur = i
            for j in xrange(points):
                res[cur] = bestval
                cur = _roll_left(cur, points)
    return res


def lbp(image, radius, points):
    '''
    features = lbp(image, radius, points)

    Compute Linear Binary Patterns

    Parameters
    ----------
        * image: input image (2-D numpy ndarray)
        * radius: radius (in pixels)
        * points: nr of points to consider

    Output
    ------
        * features: histogram of features (1-D numpy ndarray)
    

    Reference
    ---------

        Gray Scale and Rotation Invariant Texture Classification with Local Binary Patterns
            Ojala, T. Pietikainen, M. Maenpaa, T. LECTURE NOTES IN COMPUTER SCIENCE (Springer)
            2000, ISSU 1842, pages 404-420  
    '''
    image = image.astype(np.float)
    final = np.zeros(2**points)
    if points < 20:
        mapping = _precompute_mapping(points).take
    else:
        def mapping(codes):
            res = []
            for c in codes:
                bestval = cur
                cur = _roll_left(cur, points)
                if cur < bestval: bestval = cur
                res.append(bestval)
            return np.array(res)

    h,w = image.shape
    w2r = w - 2*radius

    angles = np.linspace(0, 2*np.pi, points+1)[:-1]
    coordinates = np.empty( (2, w2r, points) )
    coordinates[0] = radius * np.sin(angles)
    coordinates[1] = radius * np.cos(angles)
    coordinates1T = coordinates[1].T
    coordinates1T += np.arange(w2r)

    for row in xrange(radius, image.shape[0]-radius):
        center = image[row, radius:w-radius]
        coordinates[0] += 1
        rs = ndimage.interpolation.map_coordinates(image, coordinates, order=1)
        codes = (2**np.arange(points) * (center > rs.T).T).sum(1)
        codes = mapping(codes)
        N = len(codes)
        code = '''
        for (int i = 0; i != N; ++i) {
            ++final(codes(i));
        }
        '''
        weave.inline(code,
                ['codes','N','final'],
                type_converters=converters.blitz)
    return final

