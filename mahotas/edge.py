# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
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
from . import convolve

_hsobel_filter = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])/8.

_vsobel_filter = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]])/8.

__all__ = ['sobel']

def sobel(img, just_filter=False):
    '''
    edges = sobel(img, just_filter=Filter)

    Compute edges using Sobel's algorithm

    `edges` is a binary image of edges computed according to Sobel's algorithm.

    This implementation is tuned to match MATLAB's implementation.

    Parameters
    ----------
    img : Any 2D-ndarray
    just_filter : boolean, optional
        If true, then return the result of filtering the image with the sobel
        filters, but do not threashold.

    Returns
    -------
    edges : ndarray
        Binary image of edges, unless `just_filter`, in which case it will be
        an array of floating point values.
    '''
    # This is based on Octave's implementation,
    # but with some reverse engineering to match Matlab exactly
    img = img.astype(np.float)
    img -= img.min()
    ptp = img.ptp()
    if ptp == 0:
        return img
    img /= ptp
    # Using 'nearest' seems to be MATLAB's implementation
    vfiltered = convolve(img, _vsobel_filter, mode='nearest')
    hfiltered = convolve(img, _hsobel_filter, mode='nearest')
    filtered = vfiltered**2 + hfiltered**2
    if just_filter:
        return filtered
    thresh = 2*np.sqrt(filtered.mean())
    filtered *= (np.sqrt(filtered) < thresh)

    r,c = filtered.shape
    x = (filtered > np.hstack((np.zeros((r,1)),filtered[:,:-1]))) & (filtered > np.hstack((filtered[:,1:], np.zeros((r,1)))))
    y = (filtered > np.vstack((np.zeros(c),filtered[:-1,:]))) & (filtered > np.vstack((filtered[1:,:], np.zeros(c))))
    return x | y



