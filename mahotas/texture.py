# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
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
from . import _texture
import math

def _entropy(p):
    p = p.ravel()
    return -np.dot(np.log(p+(p==0)),p)/math.log(2.0)


def haralick(f, ignore_zeros=False, preserve_haralick_bug=False):
    '''
    feats = haralick(f, ignore_zeros=False, preserve_haralick_bug=False)

    Compute Haralick texture features

    Computes the Haralick texture features for the four 2-D directions.

    Notes
    -----
    Haralick's paper has a typo in one of the equations. This function
    implements the correct feature unless `preserve_haralick_bug` is True. The
    only reason why you'd want the buggy behaviour is if you want to match
    another implementation.

    This implementation does not compute the 14-th feature described by
    Haralick (a patch for it would be welcome).

    Parameters
    ----------
      f : A 2-D integer image.
      ignore_zeros : Whether to ignore zero pixels (default: False).
      preserve_haralick_bug : Whether to replicate Haralick's type
                              (default: False).
    Returns
    -------
      feats : A 4x13 feature vector (one row per direction).
    '''

    feats = np.zeros((4, 13), np.double)
    fm1 = f.max() + 1
    cmat = np.empty((fm1, fm1), np.long)
    k = np.arange(fm1)
    k2 = k**2
    tk = np.arange(2*fm1)
    tk2 = tk**2
    i,j = np.mgrid[:fm1,:fm1]
    ij = i*j
    i_j2_p1 = (i-j)**2
    i_j2_p1 += 1
    i_j2_p1 = 1./i_j2_p1
    i_j2_p1 = i_j2_p1.ravel()
    px_plus_y = np.empty(2*fm1, np.double)
    px_minus_y = np.empty(fm1, np.double)
    for dir in xrange(4):
        cooccurence(f, dir, cmat, symmetric=True)
        if ignore_zeros:
            cmat[1] = 0
            cmat[:,1] = 0
        T = cmat.sum()
        if not T:
            continue
        p = cmat/float(T)
        pravel = p.ravel()
        px = p.sum(0)
        py = p.sum(1)
        ux = np.dot(px, k)
        uy = np.dot(py, k)
        vx = np.dot(px, k2) - ux**2
        vy = np.dot(py, k2) - uy**2
        sx = np.sqrt(vx)
        sy = np.sqrt(vy)
        px_plus_y.fill(0)
        px_minus_y.fill(0)
        _texture.compute_plus_minus(p, px_plus_y, px_minus_y)

        feats[dir, 0] = np.dot(pravel, pravel)
        feats[dir, 1] = np.dot(k2, px_minus_y)

        feats[dir, 2] = 1./sx/sy * (np.dot(ij.ravel(), pravel) - ux*uy)

        feats[dir, 3] = vx
        feats[dir, 4] = np.dot(i_j2_p1, pravel)
        feats[dir, 5] = np.dot(tk, px_plus_y)

        feats[dir, 7] = _entropy(px_plus_y)

        # There is some confusion w.r.t. feats[dir, 6].
        #
        # Haralick's paper uses feats[dir, 7] in its computation, but it is
        # clear that feats[dir, 5] should be used (i.e., it computes a
        # variance).
        #
        if preserve_haralick_bug:
            feats[dir, 6] = ((tk-feats[dir, 7])**2*px_plus_y).sum()
        else:
            feats[dir, 6] = np.dot(tk2, px_plus_y) - feats[dir, 5]

        feats[dir,  8] = _entropy(pravel)
        feats[dir,  9] = px_minus_y.var() # This is wrongly implemented in ml_texture
        feats[dir, 10] = _entropy(px_minus_y)

        HX = _entropy(px)
        HY = _entropy(py)
        crosspxpy = np.outer(px,py)
        crosspxpy += (crosspxpy == 0) # This makes the log be zero and everything works OK below:
        crosspxpy = crosspxpy.ravel()
        HXY1 = -np.dot(pravel, np.log2(crosspxpy))
        HXY2 = _entropy(crosspxpy)

        feats[dir, 11] = (feats[dir,8]-HXY1)/max(HX,HY)
        feats[dir, 12] = np.sqrt(1-np.exp(-2.*(HXY2-feats[dir,8])))
    return feats


def cooccurence(f, direction, output=None, symmetric=True):
    '''
    cooccurence_matrix = cooccurence(f, direction, output={new matrix})

    Compute grey-level cooccurence matrix

    Parameters
    ----------
      f : An integer valued 2-D image
      direction : Direction as index into (horizontal [default], diagonal
                  [nw-se], vertical, diagonal [ne-sw])
      output : A np.long 2-D array for the result.
      symmetric : Whether return a symmetric matrix (default: False)
    Returns
    -------
      cooccurence_matrix : cooccurence matrix
    '''
    assert direction in (0,1,2,3), 'mahotas.texture.cooccurence: `direction` %s is not in range(4).' % direction
    if output is None:
        mf = f.max()
        output = np.zeros((mf+1, mf+1), np.long)
    else:
        assert np.min(output.shape) >= f.max(), 'mahotas.texture.cooccurence: output is not large enough'
        assert output.dtype == np.long, 'mahotas.texture.cooccurence: output is not of type np.long'
        output.fill(0)

    if direction == 2:
        f = f.T
    elif direction == 3:
        f = f[:, ::-1]
    diagonal = (direction in (1,3))
    _texture.cooccurence(f, output, diagonal, symmetric)
    return output

