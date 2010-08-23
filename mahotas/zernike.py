# -*- coding: utf-8 -*-
# Copyright (C) 2006-2010, Luis Pedro Coelho <lpc@cmu.edu>
# Carnegie Mellon University
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
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.

from __future__ import division
import math
import numpy as np
from center_of_mass import center_of_mass
from scipy import ndimage
from scipy import weave
from scipy.weave import converters
import _zernike

__all__ = ['zernike']

def zernike(img, D, radius, scale):
    """
    zvalues = zernike(img, D, radius, scale)

    Zernike moments through degree D

    Returns a vector of absolute Zernike moments through degree D for the
    image I.

    Parameters
    ----------
       * radius is used as the maximum radius for the Zernike polynomials.
       * scale is the scale of the image.

    Reference: Teague, MR. (1980). Image Analysis via the General
      Theory of Moments.  J. Opt. Soc. Am. 70(8):920-930.
    """
    zvalues = []

    X,Y = np.where(img > 0)
    P = img[X,Y].ravel()

# Normalize the coordinates to the center of mass and normalize
#  pixel distances using the maximum radius argument (radius)
    cofx,cofy = center_of_mass(img)
    def rescale(C, centre):
        Cn = C.astype(np.double)
        Cn -= centre
        Cn /= (radius/scale)
        return Cn.ravel()
    Xn = rescale(X, cofx)
    Yn = rescale(Y, cofy)

# Find all pixels of distance <= 1.0 to center
    k = (np.sqrt(Xn**2 + Yn**2) <= 1.)
    frac_center = np.array(P[k], np.double)/img.sum()
    Yn = Yn[k]
    Xn = Xn[k]
    frac_center = frac_center.ravel()

    for n in xrange(D+1):
        for l in xrange(n+1):
            if (n-l)%2 == 0:
                z = _zernike.znl(Xn, Yn, frac_center, float(n), float(l))
                zvalues.append(abs(z))
    return zvalues

