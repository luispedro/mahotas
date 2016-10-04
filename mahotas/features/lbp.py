# -*- coding: utf-8 -*-
# Copyright (C) 2008-2009 Robert Webb and Luis Pedro Coelho <luis@luispedro.org>
# Copyright (C) 2011-2013 Luis Pedro Coelho <luis@luispedro.org>
#
# License: MIT (see COPYING file)

import numpy as np
from ..histogram import fullhistogram

__all__ = [
    'lbp',
    'lbp_transform',
    ]

def lbp_transform(image, radius, points, ignore_zeros=False, preserve_shape=True):
    '''
    transformed = lbp(image, radius, points, ignore_zeros=False, preserve_shape=True)

    Compute Linear Binary Pattern Transform

    The return value are the transformed pixel values  **histogram** of feature counts, where position ``i``
    corresponds to the number of pixels that had code ``i``. The codes are
    compressed so that impossible codes are not used. Therefore, this is the
    ``i``th feature, not just the feature with binary code ``i``.

    Parameters
    ----------
    image : ndarray
        input image (2-D numpy ndarray)
    radius : number (integer or floating point)
        radius (in pixels)
    points : integer
        nr of points to consider
    ignore_zeros : boolean, optional
        whether to ignore zeros. Note that if you set this to ``True``, you
        will need to set ``preserve_shape`` to False. (default: False)
    preserve_shape : boolean, optional
        whether to return an array with the same shape as ``image``. (default:
        True)

    Returns
    -------
    features : 1-D numpy ndarray
        histogram of features. See above for a caveat on the interpretation of
        these.

    References
    ----------
    Gray Scale and Rotation Invariant Texture Classification with Local Binary Patterns
        Ojala, T. Pietikainen, M. Maenpaa, T. Lecture Notes in Computer Science (Springer)
        2000, ISSU 1842, pages 404-420
    '''
    from ..interpolate import shift
    from mahotas.features import _lbp

    if ignore_zeros and preserve_shape:
        raise ValueError('mahotas.features.lbp_transform: *ignore_zeros* and *preserve_shape* cannot both be used together')

    image = np.asanyarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError('mahotas.features.lbp_transform: This function is only defined for two dimensional images')

    if ignore_zeros:
        Y,X = np.nonzero(image)
        def select(im):
            return im[Y,X].ravel()
    else:
        select = np.ravel

    pixels = select(image)
    angles = np.linspace(0, 2*np.pi, points+1)[:-1]
    data = []
    for dy,dx in zip(np.sin(angles), np.cos(angles)):
        data.append(
            select(shift(image, [radius*dy,radius*dx], order=1)))
    data = np.array(data)
    codes = (data > pixels).astype(np.int32)
    codes *= (2**np.arange(points)[:,np.newaxis])
    codes = codes.sum(0)
    codes = _lbp.map(codes.astype(np.uint32), points)
    if preserve_shape:
        codes = codes.reshape(image.shape)
    return codes

def count_binary1s(array):
    '''
    one_count = count_binary1s(array)

    Count the number of 1s in the binary representation of integer values

    Definition::

        one_count.flat[i] == nr_of_1s_in_binary_representation_of(array.flat[i])

    Parameters
    ----------
    array : ndarray
        input array

    Returns
    -------
    one_count : ndarray
        output array of same type & shape as array
    '''
    from ..internal import _verify_is_integer_type
    array = np.array(array)
    _verify_is_integer_type(array, 'mahotas.features.lbp.count_binary1s')
    maxv = 1+int(np.log2(1+array.max()))
    counts = np.zeros_like(array)
    for _ in range(maxv):
        counts += (array & 1)
        array >>= 1
    return counts


def lbp(image, radius, points, ignore_zeros=False):
    '''
    features = lbp(image, radius, points, ignore_zeros=False)

    Compute Linear Binary Patterns

    The return value is a **histogram** of feature counts, where position ``i``
    corresponds to the number of pixels that had code ``i``. The codes are
    compressed so that impossible codes are not used. Therefore, this is the
    ``i``th feature, not just the feature with binary code ``i``.

    Parameters
    ----------
    image : ndarray
        input image (2-D numpy ndarray)
    radius : number (integer or floating point)
        radius (in pixels)
    points : integer
        nr of points to consider
    ignore_zeros : boolean, optional
        whether to ignore zeros (default: False)

    Returns
    -------
    features : 1-D numpy ndarray
        histogram of features. See above for a caveat on the interpretation of
        these.

    References
    ----------
    Gray Scale and Rotation Invariant Texture Classification with Local Binary Patterns
        Ojala, T. Pietikainen, M. Maenpaa, T. Lecture Notes in Computer Science (Springer)
        2000, ISSU 1842, pages 404-420
    '''
    from mahotas.features import _lbp
    codes = lbp_transform(image, radius, points, ignore_zeros=ignore_zeros, preserve_shape=False)
    final = fullhistogram(codes.astype(np.uint32))

    codes = np.arange(2**points, dtype=np.uint32)
    iters = codes.copy()
    codes = _lbp.map(codes.astype(np.uint32), points)
    pivots = (codes == iters)
    npivots = np.sum(pivots)
    compressed = final[pivots[:len(final)]]
    compressed = np.append(compressed, np.zeros(npivots - len(compressed)))
    return compressed

def lbp_names(radius, points):
    '''Return list of names (string) for LBP features

    Parameters
    ----------
    radius : number (integer or floating point)
        radius (in pixels)
    points : integer
        nr of points to consider

    Returns
    -------
    names : list of str

    See Also
    --------
    lbp : function
        Compute LBP features
    '''
    from mahotas.features import _lbp
    codes = np.arange(2**points, dtype=np.uint32)
    iters = codes.copy()
    codes = _lbp.map(codes.astype(np.uint32), points)
    pivots = (codes == iters)
    npivots = np.sum(pivots)
    return ['lbp_r{}_p{}_{}'.format(radius, points, i) for i in range(npivots)]

