# -*- coding: utf-8 -*-
# Copyright (C) 2006-2014, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np

from ..center_of_mass import center_of_mass
from . import _zernike

__all__ = ['zernike', 'zernike_moments']

def zernike(im, degree, radius, cm=None): # pragma: no cover
    """
    zvalues = zernike(im, degree, radius, cm={center_of_mass(im)})
    """
    import warnings
    warnings.warn('mahotas.zernike.zernike: This interface is deprecated. Switch the order of your arguments and use ``zernike_moments``', DeprecationWarning)
    return zernike_moments(im, radius, degree, cm)

def zernike_moments(im, radius, degree=8, cm=None):
    """
    zvalues = zernike_moments(im, radius, degree=8, cm={center_of_mass(im)})

    Zernike moments through ``degree``. These are computed on a circle of
    radius ``radius`` centered around ``cm`` (or the center of mass of the
    image, if the ``cm`` argument is not used).

    Returns a vector of absolute Zernike moments through ``degree`` for the
    image ``im``.

    Parameters
    ----------
    im : 2-ndarray
        input image
    radius : integer
        the maximum radius for the Zernike polynomials, in pixels. Note that
        the area outside the circle (centered on center of mass) defined by
        this radius is ignored.
    degree : integer, optional
        Maximum degree to use (default: 8)
    cm : pair of floats, optional
        the centre of mass to use. By default, uses the image's centre of mass.

    Returns
    -------
    zvalues : 1-ndarray of floats
        Zernike moments

    References
    ----------
    Teague, MR. (1980). Image Analysis via the General Theory of Moments.  J.
    Opt. Soc. Am. 70(8):920-930.
    """
    zvalues = []
    if cm is None:
        c0,c1 = center_of_mass(im)
    else:
        c0,c1 = cm

    Y,X = np.mgrid[:im.shape[0],:im.shape[1]]
    P = im.ravel()

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
    np.maximum(Dn, 1e-9, out=Dn)
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

    Ans = [An**p for p in range(2,degree+2)]
    Ans.insert(0, An) # An**1
    Ans.insert(0, np.ones_like(An)) # An**0
    for n in range(degree+1):
        for l in range(n+1):
            if (n-l)%2 == 0:
                z = _zernike.znl(Dn, Ans[l], frac_center, n, l)
                zvalues.append(abs(z))
    return np.array(zvalues)

