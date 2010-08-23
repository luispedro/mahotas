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

__all__ = ['zernike']

def _polar(r,theta):
    x = r * cos(theta)
    y = r * sin(theta)
    return 1*x+1j*y

_factorialtable = np.array([1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800,479001600])
def Znl(n,l,X,Y,P):
    Nelems = len(X)
    v = np.array([0 + 0.j]) # This is necessary for the C++ code to see and update v correctly
    code='''
#line 47 "zernike.py"
    using std::pow;
    using std::atan2;
    using std::polar;
    using std::conj;
    using std::complex;
    complex<double> Vnl = 0.0;
    v(0)=0;
    for (int i = 0; i != Nelems; ++i) {
        double x=X(i);
        double y=Y(i);
        double p=P(i);
        Vnl = 0.;
        for(int m = 0; m <= (n-l)/2; m++) {
            double f = (m & 1) ? -1 : 1;
            Vnl += f * _factorialtable(int(n-m)) /
                   ( _factorialtable(m) * _factorialtable((n - 2*m + l) / 2) * _factorialtable((n - 2*m - l) / 2) ) *
                   ( pow( sqrt(x*x + y*y), (double)(n - 2*m)) ) *
                   polar(1.0, l*atan2(y,x)) ;
        }
        v(0) += p * conj(Vnl);
    }
    '''
    weave.inline(code,
        ['_factorialtable','X','Y','P','v','n','l','Nelems'],
        type_converters=converters.blitz,
        compiler = 'gcc',
        headers=['<complex>'])
    v = v[0]
    v *= (n+1)/np.pi
    return v


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

# Find all non-zero pixel coordinates and values
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
                z = Znl(n,l, Xn, Yn, frac_center)
                zvalues.append(abs(z))
    return zvalues

