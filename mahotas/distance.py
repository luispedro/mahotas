# Copyright (C) 2010-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

from . import _distance
import numpy as np

__all__ = [
    'distance',
    ]

def distance(bw, metric='euclidean2'):
    '''
    dmap = distance(bw, metric='euclidean2')

    Computes the distance transform of image `bw`::

        dmap[i,j] = min_{i', j'} { (i-i')**2 + (j-j')**2 | !bw[i', j'] }

    That is, at each point, compute the distance to the background.

    If there is no background, then a very high value will be returned in all
    pixels (this is a sort of infinity).

    Parameters
    ----------
    bw : 2d-ndarray
        If boolean, ``False`` will denote the background and ``True`` the
        foreground. If not boolean, this will be interpreted as ``bw != 0``
        (this way you can use labeled images without any problems).
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
    if bw.dtype != np.bool_:
        bw = (bw != 0)
    f = np.zeros(bw.shape, np.double)
    f[bw] = len(f.shape)*max(f.shape)**2+1
    _distance.dt(f, None)
    if metric == 'euclidean':
        np.sqrt(f,f)
    return f


