# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np
from . import _texture
from ..internal import _verify_is_integer_type
import math

__all__ = ['haralick', "haralick_labels"]

def _entropy(p):
    p = p.ravel()
    return -np.dot(np.log(p+(p==0)),p)/math.log(2.0)


def haralick(f, ignore_zeros=False, preserve_haralick_bug=False, compute_14th_feature=False):
    '''
    feats = haralick(f, ignore_zeros=False, preserve_haralick_bug=False, compute_14th_feature=False)

    Compute Haralick texture features

    Computes the Haralick texture features for the four 2-D directions or
    thirteen 3-D directions (depending on the dimensions of `f`).

    Notes
    -----
    Haralick's paper has a typo in one of the equations. This function
    implements the correct feature unless `preserve_haralick_bug` is True. The
    only reason why you'd want the buggy behaviour is if you want to match
    another implementation.

    Reference
    ---------

    Cite the following reference for these features::

        @article{Haralick1973,
            author = {Haralick, Robert M. and Dinstein, Its'hak and Shanmugam, K.},
            journal = {Ieee Transactions On Systems Man And Cybernetics},
            number = {6},
            pages = {610--621},
            publisher = {IEEE},
            title = {Textural features for image classification},
            url = {http://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=4309314},
            volume = {3},
            year = {1973}
        }

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
    compute_14th_feature : bool, optional
        whether to compute & return the 14-th feature

    Returns
    -------
    feats : ndarray of np.double
        A 4x13 or 4x14 feature vector (one row per direction) if `f` is 2D,
        13x13 or 13x14 if it is 3D. The exact number of features depends on the
        value of ``compute_14th_feature``
    '''
    _verify_is_integer_type(f, 'mahotas.haralick')

    if len(f.shape) == 2:
        nr_dirs = len(_2d_deltas)
    elif len(f.shape) == 3:
        nr_dirs = len(_3d_deltas)
    else:
        raise ValueError('mahotas.texture.haralick: Can only handle 2D and 3D images.')
    feats = np.zeros((nr_dirs, 13 + bool(compute_14th_feature)), np.double)
    fm1 = f.max() + 1
    cmat = np.empty((fm1, fm1), np.int32)
    k = np.arange(fm1)
    k2 = k**2
    tk = np.arange(2*fm1)
    tk2 = tk**2
    i,j = np.mgrid[:fm1,:fm1]
    ij = i*j
    i_j2_p1 = (i - j)**2
    i_j2_p1 += 1
    i_j2_p1 = 1. / i_j2_p1
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
        p = cmat / float(T)
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

        if sx == 0. or sy == 0.:
            feats[dir, 2] = 1.
        else:
            feats[dir, 2] = (1. / sx / sy) * (np.dot(ij.ravel(), pravel) - ux * uy)

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
        feats[dir,  9] = px_minus_y.var()
        feats[dir, 10] = _entropy(px_minus_y)

        HX = _entropy(px)
        HY = _entropy(py)
        crosspxpy = np.outer(px,py)
        crosspxpy += (crosspxpy == 0) # This makes log(0) become log(1), and thus evaluate to zero, such that everything works below:
        crosspxpy = crosspxpy.ravel()
        HXY1 = -np.dot(pravel, np.log2(crosspxpy))
        HXY2 = _entropy(crosspxpy)

        if max(HX, HY) == 0.:
            feats[dir, 11] = (feats[dir,8]-HXY1)
        else:
            feats[dir, 11] = (feats[dir,8]-HXY1)/max(HX,HY)
        feats[dir, 12] = np.sqrt(1 - np.exp( -2. * (HXY2 - feats[dir,8])))

        if compute_14th_feature:
            # Square root of the second largest eigenvalue of the correlation matrix
            # Probably the faster way to do this is just SVD the whole (likely rank deficient) matrix
            # grab the second highest singular value . . . Instead, we just amputate the empty rows/cols and move on.
            nzero_rc = px != 0
            nz_pmat = p[nzero_rc,:][:,nzero_rc] # Symmetric, so this is ok!
            if nz_pmat.shape[0] > 2:
                ccm = np.corrcoef(nz_pmat)
                e_vals = np.linalg.eigvalsh(ccm)
                e_vals.sort()
                feats[dir, 13] = np.sqrt(e_vals[-2])
            else:
                feats[dir, 13] = 0

    return feats


haralick_labels = ["Angular Second Moment",
                   "Contrast",
                   "Correlation",
                   "Sum of Squares: Variance",
                   "Inverse Difference Moment",
                   "Sum Average",
                   "Sum Variance",
                   "Sum Entropy",
                   "Entropy",
                   "Difference Variance",
                   "Difference Entropy",
                   "Information Measure of Correlation 1",
                   "Information Measure of Correlation 2",
                   "Maximal Correlation Coefficient"]

_2d_deltas= [
    (0,1),
    (1,1),
    (1,0),
    (1,-1)]

_3d_deltas = [
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (1,-1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
    (1,-1, 1),
    (1, 0,-1),
    (0, 1,-1),
    (1, 1,-1),
    (1,-1,-1) ]

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

