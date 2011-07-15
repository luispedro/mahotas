# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
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
from .morph import _verify_is_integer_type
import math

__all__ = ['haralick']

def _entropy(p):
    p = p.ravel()
    return -np.dot(np.log(p+(p==0)),p)/math.log(2.0)


def haralick(f, ignore_zeros=False, preserve_haralick_bug=False):
    '''
    feats = haralick(f, ignore_zeros=False, preserve_haralick_bug=False)

    Compute Haralick texture features

    Computes the Haralick texture features for the four 2-D directions or
    thirteen 3-D directions (depending on the dimensions of `f`).

    Notes
    -----
    Haralick's paper has a typo in one of the equations. This function
    implements the correct feature unless `preserve_haralick_bug` is True. The
    only reason why you'd want the buggy behaviour is if you want to match
    another implementation.

    Bugs
    ----
    This implementation does not compute the 14-th feature described by
    Haralick (a patch for it would be welcome).

    Parameters
    ----------
    f : ndarray of integer type
        input image. 2-D and 3-D images are supported.
    ignore_zeros : bool, optional
        Whether to ignore zero pixels (default: False).
    preserve_haralick_bug : bool, optional
        whether to replicate Haralick's typo (default: False).
        You probably want to always set this to ``False`` unless you want to
        replicate someone else's wrong implementation.

    Returns
    -------
    feats : ndarray of np.double
        A 4x13 feature vector (one row per direction) if `f` is 2D, 13x13 if it
        is 3xD.
    '''
    _verify_is_integer_type(f, 'mahotas.haralick')

    if len(f.shape) == 2:
        nr_dirs = len(_2d_deltas)
    elif len(f.shape) == 3:
        nr_dirs = len(_3d_deltas)
    else:
        raise ValueError('mahotas.texture.haralick: Can only handle 2D and 3D images.')
    feats = np.zeros((nr_dirs, 13), np.double)
    fm1 = f.max() + 1
    cmat = np.empty((fm1, fm1), np.int32)
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
    for dir in xrange(nr_dirs):
        cooccurence(f, dir, cmat, symmetric=True)
        if ignore_zeros:
            cmat[0] = 0
            cmat[:,0] = 0
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
            feats[dir, 6] = np.dot(tk2, px_plus_y) - feats[dir, 5]**2

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


_2d_deltas= [(0,1), (1,1), (1,0), (1,-1)]
_3d_deltas = [
    (1,0,0),
    (1,1,0),
    (0,1,0),
    (1,-1,0),
    (0,0,1),
    (1,0,1),
    (0,1,1),
    (1,1,1),
    (1,-1,1),
    (1,0,-1),
    (0,1,-1),
    (1,1,-1),
    (1,-1,-1),
]

def cooccurence(f, direction, output=None, symmetric=True):
    '''
    cooccurence_matrix = cooccurence(f, direction, output={new matrix})

    Compute grey-level cooccurence matrix

    Parameters
    ----------
    f : ndarray of integer type
        The input image
    direction : integer
        Direction as index into (horizontal [default], diagonal
        [nw-se], vertical, diagonal [ne-sw])
    output : np.long 2 ndarray, optional
        preallocated result.
    symmetric : boolean, optional
        whether return a symmetric matrix (default: False)

    Returns
    -------
      cooccurence_matrix : cooccurence matrix
    '''
    _verify_is_integer_type(f, 'mahotas.cooccurence')
    if len(f.shape) == 2:
        assert direction in (0,1,2,3), 'mahotas.texture.cooccurence: `direction` %s is not in range(4).' % direction
    elif len(f.shape) == 3:
        assert direction in xrange(13), 'mahotas.texture.cooccurence: `direction` %s is not in range(13).' % direction
    else:
        raise ValueError('mahotas.texture.cooccurence: cannot handle images of %s dimensions.' % len(f.shape))

    if output is None:
        mf = f.max()
        output = np.zeros((mf+1, mf+1), np.int32)
    else:
        assert np.min(output.shape) >= f.max(), 'mahotas.texture.cooccurence: output is not large enough'
        assert output.dtype == np.int32, 'mahotas.texture.cooccurence: output is not of type np.int32'
        output.fill(0)

    if len(f.shape) == 2:
        Bc = np.zeros((3, 3), f.dtype)
        y,x = _2d_deltas[direction]
        Bc[y+1,x+1] = 1
    else:
        Bc = np.zeros((3, 3, 3), f.dtype)
        y,x,z = _3d_deltas[direction]
        Bc[y+1,x+1,z+1] = 1
    _texture.cooccurence(f, output, Bc, symmetric)
    return output

