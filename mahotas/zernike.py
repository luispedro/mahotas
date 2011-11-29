# -*- coding: utf-8 -*-
# Copyright (C) 2006-2011, Luis Pedro Coelho <luis@luispedro.org>
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
import numpy as np
from center_of_mass import center_of_mass
import _zernike

__all__ = ['zernike']

def zernike(img, degree, radius, cm=None):
    """
    zvalues = zernike(img, degree, radius, cm={center_of_mass(img)})

    Zernike moments through ``degree``

    Returns a vector of absolute Zernike moments through ``degree`` for the
    image ``img``.

    Parameters
    ----------
    img : 2-ndarray
        input image
    degree : integer
        Maximum degree to use
    radius : integer
        the maximum radius for the Zernike polynomials, in pixels
    cm : pair of floats, optional
        the centre of mass to use. By default, uses the image's centre of mass.

    Returns
    -------
    zvalues : 1-ndarray of floats
        Zernike moments

    Reference
    ---------
    Teague, MR. (1980). Image Analysis via the General Theory of Moments.  J.
    Opt. Soc. Am. 70(8):920-930.
    """
    zvalues = []
    if cm is None:
        c0,c1 = center_of_mass(img)
    else:
        c0,c1 = cm

    Y,X = np.mgrid[:img.shape[0],:img.shape[1]]
    P = img.ravel()

    def rescale(C, centre):
        Cn = C.astype(np.double)
        Cn -= centre
        Cn /= radius
        return Cn.ravel()
    Yn = rescale(Y, c0)
    Xn = rescale(X, c1)

    Dn = Xn**2
    Dn += Yn**2
    np.sqrt(Dn, Dn)
    k = (Dn <= 1.)
    k &= (P > 0)

    frac_center = np.array(P[k], np.double)
    frac_center = frac_center.ravel()
    frac_center /= frac_center.sum()
    Yn = Yn[k]
    Xn = Xn[k]
    Dn = Dn[k]
    An = np.empty(Yn.shape, np.complex)
    An.real = (Xn/Dn)
    An.imag = (Yn/Dn)

    Ans = [An**p for p in xrange(2,degree+2)]
    Ans.insert(0, An) # An**1
    Ans.insert(0, np.ones_like(An)) # An**0
    for n in xrange(degree+1):
        for l in xrange(n+1):
            if (n-l)%2 == 0:
                z = _zernike.znl(Dn, Ans[l], frac_center, n, l)
                zvalues.append(abs(z))
    return np.array(zvalues)

