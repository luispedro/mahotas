# Copyright (C) 2010, Luis Pedro Coelho <luis@luispedro.org>
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

from . import _distance
import numpy as np

def distance(bw, metric='euclidean2'):
    '''
    dmap = distance(bw, metric='euclidean2')

    Computes the distance transform of image `bw`::

        dmap[i,j] = min_{i', j'} { (i-i')**2 + (j-j')**2 | bw[i', j'] }

    Parameters
    ----------
    bw : 2d-ndarray
        Black & White image
    metric : str, optional
        one of 'euclidean2' (default) or 'euclidean'

    Returns
    -------
    dmap : ndarray
        distance map

    Reference
    ---------
    Felzenszwalb P, Huttenlocher D. *Distance transforms of sampled functions.
    Cornell Computing and Information.* 2004.

    Available at:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.88.1647&rep=rep1&type=pdf.
    '''

    f = np.zeros(bw.shape, np.double)
    f[bw] = len(f.shape)*max(f.shape)**2+1
    _distance.dt(f, None)
    if metric == 'euclidean':
        np.sqrt(f,f)
    return f


