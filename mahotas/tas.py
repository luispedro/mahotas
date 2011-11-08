# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# Carnegie Mellon University
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

from __future__ import division
import numpy as np
from .convolve import convolve
from .thresholding import otsu

__all__ = ['pftas', 'tas']

_M2 = np.ones((3, 3))
_M2[1, 1] = 10
_bins2 = np.arange(11)

_M3 = np.ones((3, 3, 3))
_M3[1,1,1] = _M3.sum() + 1
_bins3 = np.arange(28)

def _tas(img, thresh, margin):
    if len(img.shape) == 2:
        M = _M2
        bins = _bins2
        saved = 9
    elif len(img.shape) == 3:
        M = _M3
        bins = _bins3
        saved = 27
    else:
        raise ValueError('mahotas.tas: Cannot compute TAS for image of %s dimensions' % len(img.shape))

    def _ctas(img):
        V = convolve(img.astype(np.uint8), M)
        values,_ = np.histogram(V, bins=bins)
        values = values[:saved]
        s = values.sum()
        if s > 0:
            return values/s
        return values

    def _compute(bimg):
        alltas.append(_ctas(bimg))
        allntas.append(_ctas(~bimg))

    alltas = []
    allntas = []
    total = np.sum(img > thresh)
    mu = ((img > thresh)*img).sum() / (total + 1e-8)
    _compute( (img > mu - margin) * (img < mu + margin) )
    _compute(img > mu - margin)
    _compute(img > mu)

    return np.concatenate(alltas + allntas)

def tas(img):
    '''
    values = tas(img)

    Compute Threshold Adjacency Statistics

    TAS were presented by Hamilton et al.  in "Fast automated cell phenotype
    image classification" (http://www.biomedcentral.com/1471-2105/8/110)

    Also returns a version computed on the negative of the binarisation defined
    by Hamilton et al.

    See also pftas() for a variation without any hardcoded parameters.

    Parameters
    ----------
    img : ndarray, 2D or 3D
        input image

    Returns
    -------
    values : ndarray
        A 1-D ndarray of feature values

    See Also
    --------
    pftas : Parameter free TAS
    '''
    return _tas(img, 30, 30)

def pftas(img, T=None):
    '''
    values = pftas(img, T={mahotas.threshold.otsu(img)})

    Compute parameter free Threshold Adjacency Statistics

    TAS were presented by Hamilton et al.  in "Fast automated cell phenotype
    image classification" (http://www.biomedcentral.com/1471-2105/8/110)

    The current version is an adapted version which is free of parameters. The
    thresholding is done by using Otsu's algorithm (or can be pre-computed and
    passed in by setting `T`), the margin around the mean of pixels to be
    included is the standard deviation.

    Also returns a version computed on the negative of the binarisation defined
    by Hamilton et al.

    Use tas() to get the original version of the features.

    Parameters
    ----------
    img : ndarray, 2D or 3D
        input image
    T : integer, optional
        Threshold to use (default: compute with otsu)

    Returns
    -------
    values : ndarray
        A 1-D ndarray of feature values
    '''
    if T is None:
        T = otsu(img)
    pixels = img[img > T].ravel()
    std = pixels.std()
    return _tas(img, T, std)

