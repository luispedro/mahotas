# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT

from __future__ import division
import numpy as np

from .internal import _get_output, _verify_is_integer_type
from . import _morph

__all__ = [
        'close',
        'close_holes',
        'cwatershed',
        'cerode',
        'dilate',
        'erode',
        'get_structuring_elem',
        'hitmiss',
        'locmax',
        'locmin',
        'majority_filter'
        'open',
        'regmax',
        'regmin',
        ]

def get_structuring_elem(A,Bc):
    '''
    Bc_out = get_structuring_elem(A, Bc)

    Retrieve appropriate structuring element

    Parameters
    ----------
    A : ndarray
        array which will be operated on
    Bc : None, int, or array-like
        :None: Then Bc is taken to be 1
        :An integer: There are two associated semantics:
            connectivity
              ``Bc[y,x] = [[ is |y - 1| + |x - 1| <= Bc_i ]]``
            count
              ``Bc.sum() == Bc_i``
              This is the more traditional meaning (when one writes that
              "4-connected", this is what one has in mind).

          Fortunately, the value itself allows one to distinguish between the
          two semantics and, if used correctly, no ambiguity should ever occur.
        :An array: This should be of the same nr. of dimensions as A and will
            be passed through if of the right type. Otherwise, it will be cast.

    Returns
    -------
    Bc_out : ndarray
        Structuring element. This array will be of the same type as A,
        C-contiguous.

    '''
    translate_sizes = {
            (2, 4) : 1,
            (2, 8) : 2,
            (3, 6) : 1,
    }
    if Bc is None:
        Bc = 1
    elif type(Bc) == int and (len(A.shape), Bc) in translate_sizes:
        Bc = translate_sizes[len(A.shape),Bc]
    elif type(Bc) != int:
        if len(A.shape) != len(Bc.shape):
            raise ValueError('morph.get_structuring_elem: Bc does not have the correct number of dimensions.')
        Bc = np.asanyarray(Bc, A.dtype)
        if not Bc.flags.contiguous:
            return Bc.copy()
        return Bc

    # Special case typical case:
    if len(A.shape) == 2 and Bc == 1:
        return np.array([
                [0,1,0],
                [1,1,1],
                [0,1,0]], dtype=A.dtype)
    max1 = Bc
    Bc = np.zeros((3,)*len(A.shape), dtype=A.dtype)
    centre = np.ones(len(A.shape))
    # This is pretty slow, but this should be a tiny array, so who cares
    for i in xrange(Bc.size):
        pos = np.unravel_index(i, Bc.shape)
        pos -= centre
        if np.sum(np.abs(pos)) <= max1:
            Bc.flat[i] = 1
    return Bc

def dilate(A, Bc=None, out=None, output=None):
    '''
    dilated = dilate(A, Bc={3x3 cross}, out={np.empty_like(A)})

    Morphological dilation.

    The type of operation depends on the ``dtype`` of ``A``! If boolean, then
    the dilation is binary, else it is greyscale dilation. In the case of
    greyscale dilation, the smallest value in the domain of ``Bc`` is
    interpreted as +Inf.

    Parameters
    ----------
    A : ndarray of bools
        input array
    Bc : ndarray, optional
        Structuring element. By default, use a cross (see
        ``get_structuring_elem`` for details on the default).

    Returns
    -------
    dilated : ndarray
        dilated version of ``A``

    See Also
    --------
    erode
    '''
    _verify_is_integer_type(A, 'dilate')
    Bc = get_structuring_elem(A,Bc)
    output = _get_output(A, out, 'dilate', output=output)
    return _morph.dilate(A, Bc, output)

def erode(A, Bc=None, out=None, output=None):
    '''
    eroded = erode(A, Bc={3x3 cross}, out={np.empty_as(A)})

    Morphological erosion.

    The type of operation depends on the ``dtype`` of ``A``! If boolean, then
    the erosion is binary, else it is greyscale erosion. In the case of
    greyscale erosion, the smallest value in the domain of ``Bc`` is
    interpreted as -Inf.

    Parameters
    ----------
    A : ndarray
        input image
    Bc : ndarray, optional
        Structuring element. By default, use a cross (see
        ``get_structuring_elem`` for details on the default).

    Returns
    -------
    erosion : ndarray
        eroded version of ``A``

    See Also
    --------
    dilate
    '''
    _verify_is_integer_type(A,'erode')
    Bc = get_structuring_elem(A,Bc)
    output = _get_output(A, out, 'erode', output=output)
    return _morph.erode(A, Bc, output)


def cerode(f, g, Bc=None, output=None):
    '''
    conditionally_eroded = erode(f, g, Bc={3x3 cross}, output={np.empty_as(A)})

    Conditional morphological erosion.

    The type of operation depends on the ``dtype`` of ``A``! If boolean, then
    the erosion is binary, else it is greyscale erosion. In the case of
    greyscale erosion, the smallest value in the domain of ``Bc`` is
    interpreted as -Inf.

    Parameters
    ----------
    f : ndarray
        input image
    g : ndarray
        conditional image
    Bc : ndarray, optional
        Structuring element. By default, use a cross (see
        ``get_structuring_elem`` for details on the default).

    Returns
    -------
    conditionally_eroded : ndarray
        eroded version of ``f`` conditioned on ``g``

    See Also
    --------
    erode : function
        Unconditional version of this function
    dilate
    '''
    f = np.maximum(f, g)
    _verify_is_integer_type(f, 'cerode')
    Bc = get_structuring_elem(f, Bc)
    output = _get_output(f, output, 'cerode')
    f = _morph.erode(f, Bc, output)
    return np.maximum(f, g, out=f)

def cwatershed(surface, markers, Bc=None, return_lines=False):
    '''
    W = cwatershed(surface, markers, Bc=None, return_lines=False)
    W,WL = cwatershed(surface, markers, Bc=None, return_lines=True)

    Seeded Watershed

    Parameters
    ----------
    surface : image
    markers : image
        initial markers (must be a labeled image)
    Bc : ndarray, optional
        structuring element (default: 3x3 cross)
    return_lines : boolean, optional
        whether to return separating lines (in addition to regions)

    Returns
    -------
    W : Regions image (i.e., W[i,j] == region for pixel (i,j))
    WL : Lines image (`if return_lines==True`)
    '''
    _verify_is_integer_type(surface, 'cwatershed')
    _verify_is_integer_type(markers, 'cwatershed')
    if surface.shape != markers.shape:
        raise ValueError('morph.cwatershed: Markers array should have the same shape as value array.')
    if markers.dtype != surface.dtype:
        markers = markers.astype(surface.dtype)
    Bc = get_structuring_elem(surface, Bc)
    return _morph.cwatershed(surface, markers, Bc, bool(return_lines))

def hitmiss(input, Bc, out=None, output=None):
    '''
    filtered = hitmiss(input, Bc, out=np.zeros_like(input))

    Hit & Miss transform

    For a given pixel position, the hit&miss is ``True`` if, when ``Bc`` is
    overlaid on ``input``, centered at that position, the ``1`` values line up
    with ``1``s, while the ``0``s line up with ``0``s (``2``s correspond to
    *don't care*).

    Example
    -------

    ::

        print hitmiss(np.array([
                    [0,0,0,0,0],
                    [0,1,1,1,1],
                    [0,0,1,1,1]]),
                np.array([
                    [0,0,0],
                    [2,1,1],
                    [2,1,1]]))

        prints::

            [[0 0 0 0 0]
             [0 0 1 1 0]
             [0 0 0 0 0]]



    Parameters
    ----------
    input : input ndarray
        This is interpreted as a binary array.
    Bc : ndarray
        hit & miss template, values must be one of (0, 1, 2)
    out : output array

    Returns
    -------
    filtered : ndarray
    '''
    _verify_is_integer_type(input, 'hitmiss')
    _verify_is_integer_type(Bc, 'hitmiss')
    if input.dtype != Bc.dtype:
        if input.dtype == np.bool_:
            input = input.view(np.uint8)
            if Bc.dtype == np.bool_:
                Bc = Bc.view(np.uint8)
            else:
                Bc = Bc.astype(np.uint8)
        else:
            Bc = Bc.astype(input.dtype)
    
    if out is None and output is not None:
        out = output

    if out is None:
        out = np.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError('mahotas.hitmiss: out must be of same shape as input')
        if out.dtype != input.dtype:
            if out.dtype == np.bool_ and input.dtype == np.uint8:
                out = out.view(np.uint8)
            else:
                raise TypeError('mahotas.hitmiss: out must be of same type as input')
    return _morph.hitmiss(input, Bc, out)


def open(f, Bc=None, out=None, output=None):
    """
    y = open(f, Bc={3x3 cross}, out={np.empty_like(f)})

    Morphological opening.

    `open` creates the image y by the morphological opening of the
    image `f` by the structuring element `Bc`.

    In the binary case, the opening by the structuring element `Bc` may be
    interpreted as the union of translations of `b` included in `f`. In the
    gray-scale case, there is a similar interpretation taking the functions
    umbra.

    Parameters
    ----------
    f : ndarray
        Gray-scale (uint8 or uint16) or binary image.
    Bc : ndarray, optional
        Structuring element (default: 3x3 elementary cross).
    out : ndarray, optional
        Output array

    Returns
    -------
    y : ndarray

    See Also
    --------
    open : function
    """
    _verify_is_integer_type(f, 'open')
    Bc = get_structuring_elem(f, Bc)
    eroded = erode(f, Bc, out=out)
    # We need to copy for the simple reason that otherwise, the image will be
    # modified in place, which can mess up the implementation
    return dilate(eroded.copy(), Bc, out=eroded)


def close(f, Bc=None, out=None, output=None):
    """
    y = close(f, Bc={3x3 cross}, out={np.empty_like(f)})

    Morphological closing.

    `close` creates the image `y` by the morphological closing of the
    image `f` by the structuring element `Bc`. In the binary case, the
    closing by a structuring element `Bc` may be interpreted as the
    intersection of all the binary images that contain the image `f`
    and have a hole equal to a translation of `Bc`. In the gray-scale
    case, there is a similar interpretation taking the functions
    umbra.

    Parameters
    ----------
    f : ndarray
        Gray-scale (uint8 or uint16) or binary image.
    Bc : ndarray, optional
        Structuring element. (Default: 3x3 elementary cross).
    out : ndarray, optional
        Output array

    Returns
    -------
    y : ndarray

    See Also
    --------
    open : function
    """
    _verify_is_integer_type(f, 'close')
    Bc = get_structuring_elem(f, Bc)
    dilated = dilate(f, Bc, out=out)
    # We need to copy for the simple reason that otherwise, the image will be
    # modified in place, which can mess up the implementation
    return erode(dilated.copy(), Bc, out=dilated)


def close_holes(ref, Bc=None):
    '''
    closed = close_holes(ref, Bc=None):

    Close Holes

    Parameters
    ----------
    ref : ndarray
        Reference image. This should be a binary image.
    Bc : structuring element, optional
        Default: 3x3 cross

    Returns
    -------
    closed : ndarray
        superset of `ref` (i.e. with closed holes)
    '''
    ref = np.ascontiguousarray(ref, dtype=np.bool_)
    Bc = get_structuring_elem(ref, Bc)
    return _morph.close_holes(ref, Bc)


def majority_filter(img, N=3, out=None, output=None):
    '''
    filtered = majority_filter(img, N=3, out={np.empty(img.shape, np.bool)})

    Majority filter

    filtered[y,x] is positive if the majority of pixels in the squared of size
    `N` centred on (y,x) are positive.

    Parameters
    ----------
    img : ndarray
        input img (currently only 2-D images accepted)
    N : int, optional
        size of filter (must be odd integer), defaults to 3.
    out : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as `img`

    Returns
    -------
    filtered : ndarray
        boolean image of same size as img.
    '''
    img = img.astype(np.bool_)
    output = _get_output(img, out, 'majority_filter', np.bool_, output=output)
    if N <= 1:
        raise ValueError('mahotas.majority_filter: filter size must be positive')
    if not N&1:
        import warnings
        warnings.warn('mahotas.majority_filter: size argument must be odd. Adding 1.')
        N += 1
    return _morph.majority_filter(img, N, output)


def _remove_centre(Bc):
    index = [s//2 for s in Bc.shape]
    Bc[tuple(index)] = False
    return Bc

def locmax(f, Bc=None, out=None, output=None):
    '''
    filtered = locmax(f, Bc={3x3 cross}, out={np.empty(f.shape, bool)})

    Local maxima

    Parameters
    ----------
    f : ndarray
    Bc : ndarray, optional
        structuring element
    out : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as `f`

    Returns
    -------
    filtered : ndarray
        boolean image of same size as f.

    See Also
    --------
    locmin : function
        Regional minima
    '''
    _verify_is_integer_type(f, 'locmax')
    Bc = get_structuring_elem(f, Bc)
    output = _get_output(f, out, 'locmax', np.bool_, output=output)
    Bc = _remove_centre(Bc.copy())
    return _morph.locmin_max(f, Bc, output, False)


def locmin(f, Bc=None, output=None):
    '''
    filtered = locmin(f, Bc={3x3 cross}, output={np.empty(f.shape, bool)})

    Local minima

    Parameters
    ----------
    f : ndarray
    Bc : ndarray, optional
        structuring element
    output : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as `f`

    Returns
    -------
    filtered : ndarray
        boolean image of same size as f.

    See Also
    --------
    locmax : function
        Regional maxima
    '''
    _verify_is_integer_type(f, 'locmin')
    Bc = get_structuring_elem(f, Bc)
    Bc = _remove_centre(Bc.copy())
    output = _get_output(f, output, 'locmin', np.bool_)
    return _morph.locmin_max(f, Bc, output, True)


def regmin(f, Bc=None, out=None, output=None):
    '''
    filtered = regmin(f, Bc={3x3 cross}, out={np.empty(f.shape, bool)})

    Regional minima

    Parameters
    ----------
    f : ndarray
    Bc : ndarray, optional
        structuring element
    out : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as `f`

    Returns
    -------
    filtered : ndarray
        boolean image of same size as f.

    See Also
    --------
    locmin : function
        Local minima
    '''
    _verify_is_integer_type(f, 'regmin')
    Bc = get_structuring_elem(f, Bc)
    Bc = _remove_centre(Bc.copy())
    output = _get_output(f, out, 'regmin', np.bool_, output=output)
    return _morph.regmin_max(f, Bc, output, True)


def regmax(f, Bc=None, out=None, output=None):
    '''
    filtered = regmax(f, Bc={3x3 cross}, out={np.empty(f.shape, bool)})

    Regional maxima

    Parameters
    ----------
    f : ndarray
    Bc : ndarray, optional
        structuring element
    out : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as `f`

    Returns
    -------
    filtered : ndarray
        boolean image of same size as f.

    See Also
    --------
    locmax : function
        Local maxima
    '''
    _verify_is_integer_type(f, 'regmax')
    Bc = get_structuring_elem(f, Bc)
    Bc = _remove_centre(Bc.copy())
    output = _get_output(f, out, 'regmax', np.bool_, output=output)
    return _morph.regmin_max(f, Bc, output, False)
