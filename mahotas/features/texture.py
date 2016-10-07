# Copyright (C) 2008-2016, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING file)


import numpy as np
from . import _texture
from ..internal import _verify_is_integer_type

__all__ = [
    'haralick',
    'haralick_labels',
    'cooccurence',
    ]

def _entropy(p):
    p = p.ravel()
    p1 = p.copy()
    p1 += (p==0)
    return -np.dot(np.log2(p1), p)


def haralick(f,
            ignore_zeros=False,
            preserve_haralick_bug=False,
            compute_14th_feature=False,
            return_mean=False,
            return_mean_ptp=False,
            use_x_minus_y_variance=False,
            distance=1
            ):
    '''
    feats = haralick(f,
            ignore_zeros=False,
            preserve_haralick_bug=False,
            compute_14th_feature=False,
            return_mean=False,
            return_mean_ptp=False,
            use_x_minus_y_variance=False,
            distance=1
            )

    Compute Haralick texture features

    Computes the Haralick texture features for the four 2-D directions or
    thirteen 3-D directions (depending on the dimensions of `f`).

    ``ignore_zeros`` can be used to have the function ignore any zero-valued
    pixels (as background). If there are no-nonzero neighbour pairs in all
    directions, an exception is raised. Note that this can happen even with
    some non-zero pixels, e.g.::

         0 0 0 0
         0 1 0 0
         0 1 0 0
         0 0 0 0

    would trigger an error when ``ignore_zeros=True`` as there are no
    horizontal non-zero pairs!

    Notes
    -----
    Haralick's paper has a typo in one of the equations. This function
    implements the correct feature unless `preserve_haralick_bug` is True. The
    only reason why you'd want the buggy behaviour is if you want to match
    another implementation.

    References
    ----------

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
    distance: int, optional (default=1)
        The distance to consider while computing the cooccurence matrix.
    ignore_zeros : bool, optional
        Whether to ignore zero pixels (default: False).

    Other Parameters
    ----------------
    preserve_haralick_bug : bool, optional
        whether to replicate Haralick's typo (default: False).
        You probably want to always set this to ``False`` unless you want to
        replicate someone else's wrong implementation.
    compute_14th_feature : bool, optional
        whether to compute & return the 14-th feature
    return_mean : bool, optional
        When set, the function returns the mean across all the directions
        (default: False).
    return_mean_ptp : bool, optional
        When set, the function returns the mean and ptp (point-to-point, i.e.,
        difference between max() and min()) across all the directions (default:
        False).
    use_x_minus_y_variance : bool, optional
        Feature 10 (index 9) has two interpretations, as the variance of \|x-y\|
        or as the variance of P(\|x-y\|). In order to achieve compatibility with
        other software and previous versions of mahotas, mahotas defaults to
        using ``VAR[P(\|x-y\|)]``; if this argument is True, then it uses
        ``VAR[\|x-y\|]`` (default: False)


    Returns
    -------
    feats : ndarray of np.double
        A 4x13 or 4x14 feature vector (one row per direction) if `f` is 2D,
        13x13 or 13x14 if it is 3D. The exact number of features depends on the
        value of ``compute_14th_feature`` Also, if either ``return_mean`` or
        ``return_mean_ptp`` is set, then a single dimensional array is
        returned.
    '''
    _verify_is_integer_type(f, 'mahotas.haralick')

    if len(f.shape) == 2:
        nr_dirs = len(_2d_deltas)
    elif len(f.shape) == 3:
        nr_dirs = len(_3d_deltas)
    else:
        raise ValueError('mahotas.texture.haralick: Can only handle 2D and 3D images.')
    fm1 = f.max() + 1
    cmat = np.empty((fm1, fm1), np.int32)
    def all_cmatrices():
        for dir in range(nr_dirs):
            cooccurence(f, dir, cmat, symmetric=True, distance=distance)
            yield cmat
    return haralick_features(all_cmatrices(),
                        ignore_zeros=ignore_zeros,
                        preserve_haralick_bug=preserve_haralick_bug,
                        compute_14th_feature=compute_14th_feature,
                        return_mean=return_mean,
                        return_mean_ptp=return_mean_ptp,
                        use_x_minus_y_variance=use_x_minus_y_variance,
                        )

def haralick_features(cmats,
                    ignore_zeros=False,
                    preserve_haralick_bug=False,
                    compute_14th_feature=False,
                    return_mean=False,
                    return_mean_ptp=False,
                    use_x_minus_y_variance=False,
                    ):
    '''
    features = haralick_features(cmats,
                    ignore_zeros=False,
                    preserve_haralick_bug=False,
                    compute_14th_feature=False,
                    return_mean=False,
                    return_mean_ptp=False,
                    use_x_minus_y_variance=False,
                    )

    Computers Haralick features for the given cooccurrence matrices.

    This function is not usually necessary, as you can call ``haralick`` with
    an image to obtain features for that image. Use only if you know what you
    are doing.

    Notes
    -----
    Haralick's paper has a typo in one of the equations. This function
    implements the correct feature unless `preserve_haralick_bug` is True. The
    only reason why you'd want the buggy behaviour is if you want to match
    another implementation.

    References
    ----------

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
    cmats : sequence of ndarrays
        This should be a sequence of ndarrays, all square and all of the same
        shape.
    ignore_zeros : bool, optional
        Whether to ignore zero pixels (default: False).

    Other Parameters
    ----------------
    preserve_haralick_bug : bool, optional
        whether to replicate Haralick's typo (default: False).
        You probably want to always set this to ``False`` unless you want to
        replicate someone else's wrong implementation.
    compute_14th_feature : bool, optional
        whether to compute & return the 14-th feature
    return_mean : bool, optional
        When set, the function returns the mean across all the directions
        (default: False).
    return_mean_ptp : bool, optional
        When set, the function returns the mean and ptp (point-to-point, i.e.,
        difference between max() and min()) across all the directions (default:
        False).
    use_x_minus_y_variance : bool, optional
        Feature 10 (index 9) has two interpretations, as the variance of |x-y|
        or as the variance of P(|x-y|). In order to achieve compatibility with
        other software and previous versions of mahotas, mahotas defaults to
        using ``VAR[P(|x-y|)]``; if this argument is True, then it uses
        ``VAR[|x-y|]`` (default: False)

    Returns
    -------
    feats : ndarray of np.double
        A 4x13 or 4x14 feature vector (one row per direction) if `f` is 2D,
        13x13 or 13x14 if it is 3D. The exact number of features depends on the
        value of ``compute_14th_feature`` Also, if either ``return_mean`` or
        ``return_mean_ptp`` is set, then a single dimensional array is
        returned.

    See Also
    --------
    haralick : function
        compute Haralick features for an image
    '''
    if return_mean and return_mean_ptp:
        raise ValueError("mahotas.haralick_features: Cannot set both `return_mean` and `return_mean_ptp`")
    features = []
    for cmat in cmats:
        feats = np.zeros(13 + bool(compute_14th_feature), np.double)
        if ignore_zeros:
            cmat[0] = 0
            cmat[:,0] = 0
        T = cmat.sum()
        if not T:
            raise ValueError('mahotas.haralick_features: the input is empty. Cannot compute features!\n' +
                                'This can happen if you are using `ignore_zeros`' )
        if not len(features):
            maxv = len(cmat)
            k = np.arange(maxv)
            k2 = k**2
            tk = np.arange(2*maxv)
            tk2 = tk**2
            i,j = np.mgrid[:maxv,:maxv]
            ij = i*j
            i_j2_p1 = (i - j)**2
            i_j2_p1 += 1
            i_j2_p1 = 1. / i_j2_p1
            i_j2_p1 = i_j2_p1.ravel()
            px_plus_y = np.empty(2*maxv, np.double)
            px_minus_y = np.empty(maxv, np.double)
        elif maxv != len(cmat):
            raise ValueError('mahotas.haralick_features: All cmatrices must be of the same size')

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

        feats[0] = np.dot(pravel, pravel)
        feats[1] = np.dot(k2, px_minus_y)

        if sx == 0. or sy == 0.:
            feats[2] = 1.
        else:
            feats[2] = (1. / sx / sy) * (np.dot(ij.ravel(), pravel) - ux * uy)

        feats[3] = vx
        feats[4] = np.dot(i_j2_p1, pravel)
        feats[5] = np.dot(tk, px_plus_y)

        feats[7] = _entropy(px_plus_y)

        # There is some confusion w.r.t. feats[6].
        #
        # Haralick's paper uses feats[7] in its computation, but it is
        # clear that feats[5] should be used (i.e., it computes a
        # variance).
        #
        if preserve_haralick_bug:
            feats[6] = ((tk-feats[7])**2*px_plus_y).sum()
        else:
            feats[6] = np.dot(tk2, px_plus_y) - feats[5]**2

        feats[ 8] = _entropy(pravel)
        if use_x_minus_y_variance:
            mu_x_minus_y = np.dot(px_minus_y, k)
            mu_x_minus_y_sq = np.dot(px_minus_y, k2)
            feats[ 9] = mu_x_minus_y_sq - mu_x_minus_y**2
        else:
            feats[ 9] = px_minus_y.var()
        feats[10] = _entropy(px_minus_y)

        HX = _entropy(px)
        HY = _entropy(py)
        crosspxpy = np.outer(px,py)
        crosspxpy += (crosspxpy == 0) # This makes log(0) become log(1), and thus evaluate to zero, such that everything works below:
        crosspxpy = crosspxpy.ravel()
        HXY1 = -np.dot(pravel, np.log2(crosspxpy))
        HXY2 = _entropy(crosspxpy)

        if max(HX, HY) == 0.:
            feats[11] = (feats[8]-HXY1)
        else:
            feats[11] = (feats[8]-HXY1)/max(HX,HY)
        feats[12] = np.sqrt(max(0,1 - np.exp( -2. * (HXY2 - feats[8]))))

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
                feats[13] = np.sqrt(e_vals[-2])
            else:
                feats[13] = 0
        features.append(feats)

    features = np.array(features)
    if return_mean:
        return features.mean(axis=0)
    if return_mean_ptp:
        mean = features.mean(axis=0)
        ptp = features.ptp(axis=0)
        return np.concatenate((mean,ptp))

    return features


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

def cooccurence(f, direction, output=None, symmetric=True, distance=1):
    '''
    cooccurence_matrix = cooccurence(f,
                            direction,
                            output={new matrix},
                            symmetric=True,
                            distance=1)

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
        whether return a symmetric matrix (default: True)
    distance : integer, optional (default=1)
        Distance to the central pixel to consider.

    Returns
    -------
      cooccurence_matrix : cooccurence matrix
    '''
    _verify_is_integer_type(f, 'mahotas.cooccurence')
    if len(f.shape) == 2 and not (0 <= direction < 4):
        raise ValueError('mahotas.texture.cooccurence: `direction` {0} is not in range(4).'.format(direction))
    elif len(f.shape) == 3 and not (0 <= direction < 13):
        raise ValueError('mahotas.texture.cooccurence: `direction` {0} is not in range(13).'.format(direction))
    elif len(f.shape) not in (2,3):
        raise ValueError('mahotas.texture.cooccurence: cannot handle images of %s dimensions.' % len(f.shape))

    if output is None:
        mf = f.max()
        output = np.zeros((mf+1, mf+1), np.int32)
    else:
        assert np.min(output.shape) >= f.max(), 'mahotas.texture.cooccurence: output is not large enough'
        assert output.dtype == np.int32, 'mahotas.texture.cooccurence: output is not of type np.int32'
        output.fill(0)

    if len(f.shape) == 2:
        mask_size = 2 * distance + 1
        Bc = np.zeros((mask_size, mask_size), f.dtype)
        y, x = tuple(distance * i for i in _2d_deltas[direction])
        Bc[y + distance, x + distance] = 1
    else:
        mask_size = 2 * distance + 1
        Bc = np.zeros((mask_size, mask_size, mask_size), f.dtype)
        y, x, z = tuple(distance * i for i in _3d_deltas[direction])
        Bc[y + distance, x + distance, z + distance] = 1
    _texture.cooccurence(f, output, Bc, symmetric)
    return output

