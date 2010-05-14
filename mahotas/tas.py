# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
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
from scipy import ndimage
from mahotas.thresholding import otsu

__all__ = ['pftas', 'tas']

_M = np.array([
    [1,  1, 1],
    [1, 10, 1],
    [1,  1, 1]
    ])
_bins = np.arange(11)

def _tas(img, thresh, margin):
    def _ctas(img):
        V = ndimage.convolve(img.astype(np.uint8), _M)
        values,_ = np.histogram(V, bins=_bins)
        values = values[:9]
        return values/values.sum()

    def _compute(bimg):
        alltas.append(_ctas(bimg))
        allntas.append(_ctas(~bimg))

    alltas = []
    allntas = []
    mu = ((img > thresh)*img).sum() / (img > thresh).sum()
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
      img : A 2-D image
    Returns
    -------
      values : A 1-D ndarray of feature values
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
      img : A 2-D image
      T : Threshold to use (default: compute with otsu)
    Returns
    -------
      values : A 1-D ndarray of feature values
    '''
    if T is None:
        T = otsu(img)
    pixels = img[img > T].ravel()
    std = pixels.std()
    return _tas(img, T, std)

