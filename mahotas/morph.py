# Copyright (C) 2008-2014, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT

from __future__ import division
import numpy as np

from .internal import _get_output, _verify_is_integer_type, _check_2
from . import _morph

__all__ = [
        'close',
        'close_holes',
        'cwatershed',
        'cerode',
        'dilate',
        'disk',
        'cdilate',
        'erode',
        'get_structuring_elem',
        'hitmiss',
        'locmax',
        'locmin',
        'majority_filter',
        'open',
        'regmax',
        'regmin',
        'tophat_open',
        'tophat_close',
        'subm',
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
        if A.ndim != Bc.ndim:
            raise ValueError('morph.get_structuring_elem: Bc does not have the correct number of dimensions. [array has {} coordinates; Bc has {}.]'.format(A.ndim, Bc.ndim))
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
    for i in range(Bc.size):
        pos = np.unravel_index(i, Bc.shape)
        pos -= centre
        if np.sum(np.abs(pos)) <= max1:
            Bc.flat[i] = 1
    return Bc

def disk(radius, dim=2):
    '''
    D = disk(radius, dim=2)

    Return a binary disk structuring element of radius ``radius`` and dimension ``dim``

    Parameters
    ----------
    radius : int
        Radius (in pixels) of returned disk
    dim : int, optional
        Dimension of returned array (default: 2)

    Returns
    -------
    D : boolean ndarray
    '''
    import numpy as np
    if dim <= 0:
        raise ValueError('mahotas.morph.disk: dimension must be positive')
    shape = [(radius*2+1) for _ in range(dim)]
    if dim == 2:
        return _morph.disk_2d(np.zeros(shape, bool), radius)
    indices = np.indices(shape, float)
    indices -= radius
    indices **= 2
    return (indices.sum(0) < (radius**2))

def dilate(A, Bc=None, out=None, output=None):
    '''
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
    out : ndarray, optional
        output array. If used, this must be a C-array of the same ``dtype`` as
        ``A``. Otherwise, a new array is allocated.
    output : deprecated
        Do not use

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
    out : ndarray, optional
        output array. If used, this must be a C-array of the same ``dtype`` as
        ``A``. Otherwise, a new array is allocated.

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


def cerode(f, g, Bc=None, out=None, output=None):
    '''
    conditionally_eroded = cerode(f, g, Bc={3x3 cross}, out={np.empty_as(A)})

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
    out = _get_output(f, out, 'cerode', output=output)
    f = _morph.erode(f, Bc, out)
    return np.maximum(f, g, out=f)

def cdilate(f, g, Bc=None, n=1):
    """
    y = cdilate(f, g, Bc={3x3 cross}, n=1)

    Conditional dilation

    `cdilate` creates the image `y` by dilating the image `f` by the
    structuring element `Bc` conditionally to the image `g`. This
    operator may be applied recursively `n` times.

    Parameters
    ----------
    f : Gray-scale (uint8 or uint16) or binary image.
    g : Conditioning image. (Gray-scale or binary).
    Bc : Structuring element (default: 3x3 cross)
    n : Number of iterations (default: 1)

    Returns
    -------
    y : Image
    """
    _verify_is_integer_type(f, 'cdilate')
    Bc = get_structuring_elem(f, Bc)
    f = np.minimum(f, g)
    for i in range(n):
        prev = f
        f = dilate(f, Bc)
        f = np.minimum(f, g)
        if np.all(f == prev):
            break
    return f


def cwatershed(surface, markers, Bc=None, return_lines=False):
    '''
    W = cwatershed(surface, markers, Bc=None, return_lines=False)
    W,WL = cwatershed(surface, markers, Bc=None, return_lines=True)

    Seeded watershed in n-dimensions

    This function computes the watershed transform on the input surface (which
    may actually be an n-dimensional volume).

    This function requires initial seed points. A traditional way of
    initializing watershed is to use regional minima::

        minima = mh.regmin(f)
        markers,nr_markers = mh.label(minima)
        W = cwatershed(f, minima)

    Parameters
    ----------
    surface : image
    markers : image
        initial markers (must be a labeled image, i.e., one where 0 represents
        the background and higher integers represent different regions)
    Bc : ndarray, optional
        structuring element (default: 3x3 cross)
    return_lines : boolean, optional
        whether to return separating lines (in addition to regions)

    Returns
    -------
    W : integer ndarray (int64 ints)
        Regions image (i.e., W[i,j] == region for pixel (i,j))
    WL : Lines image (`if return_lines==True`)
    '''
    _verify_is_integer_type(markers, 'cwatershed')
    if surface.shape != markers.shape:
        raise ValueError('morph.cwatershed: Markers array should have the same shape as value array.')
    markers = markers.astype(np.int64)
    Bc = get_structuring_elem(surface, Bc)
    return _morph.cwatershed(surface, markers, Bc, bool(return_lines))

def hitmiss(input, Bc, out=None, output=None):
    '''
    filtered = hitmiss(input, Bc, out=np.zeros_like(input))

    Hit & Miss transform

    For a given pixel position, the hit&miss is ``True`` if, when ``Bc`` is
    overlaid on ``input``, centered at that position, the ``1`` values line up
    with ``1``\ s, while the ``0``\ s line up with ``0``\ s (``2``\ s correspond to
    *don't care*).

    Examples
    --------

    ::

        print(hitmiss(np.array([
                    [0,0,0,0,0],
                    [0,1,1,1,1],
                    [0,0,1,1,1]]),
                np.array([
                    [0,0,0],
                    [2,1,1],
                    [2,1,1]])))

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
    out : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as ``input``
    output : deprecated
        Do not use

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
    output : deprecated
        Do not use

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
    output : deprecated
        Do not use

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
    _check_2(ref, 'close_holes')
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
    output : deprecated
        Do not use

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
    output : deprecated
        Do not use

    Returns
    -------
    filtered : ndarray
        boolean image of same size as f.

    See Also
    --------
    regmax : function
        Regional maxima. This is a stricter criterion than the local maxima as
        it takes the whole object into account and not just the neighbourhood
        defined by ``Bc``::

            0 0 0 0 0
            0 0 2 0 0
            0 0 2 0 0
            0 0 3 0 0
            0 0 3 0 0
            0 0 0 0 0

        The top 2 is a local maximum because it has the maximal value in its
        neighbourhood, but it is not a regional maximum.

    locmin : function
        Local minima
    '''
    Bc = get_structuring_elem(f, Bc)
    output = _get_output(f, out, 'locmax', np.bool_, output=output)
    Bc = _remove_centre(Bc.copy())
    return _morph.locmin_max(f, Bc, output, False)


def locmin(f, Bc=None, out=None, output=None):
    '''
    filtered = locmin(f, Bc={3x3 cross}, out={np.empty(f.shape, bool)})

    Local minima

    Parameters
    ----------
    f : ndarray
    Bc : ndarray, optional
        structuring element
    out : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as `f`
    output : deprecated
        Do not use

    Returns
    -------
    filtered : ndarray
        boolean image of same size as f.

    See Also
    --------
    locmax : function
        Regional maxima
    '''
    Bc = get_structuring_elem(f, Bc)
    Bc = _remove_centre(Bc.copy())
    output = _get_output(f, out, 'locmin', np.bool_, output=output)
    return _morph.locmin_max(f, Bc, output, True)


def regmin(f, Bc=None, out=None, output=None):
    '''
    filtered = regmin(f, Bc={3x3 cross}, out={np.empty(f.shape, bool)})

    Regional minima. See the documentation for ``regmax`` for more details.

    Parameters
    ----------
    f : ndarray
    Bc : ndarray, optional
        structuring element
    out : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as `f`
    output : deprecated
        Do not use

    Returns
    -------
    filtered : ndarray
        boolean image of same size as f.

    See Also
    --------
    locmin : function
        Local minima
    '''
    Bc = get_structuring_elem(f, Bc)
    Bc = _remove_centre(Bc.copy())
    output = _get_output(f, out, 'regmin', np.bool_, output=output)
    return _morph.regmin_max(f, Bc, output, True)


def regmax(f, Bc=None, out=None, output=None):
    '''
    filtered = regmax(f, Bc={3x3 cross}, out={np.empty(f.shape, bool)})

    Regional maxima. This is a stricter criterion than the local maxima as
    it takes the whole object into account and not just the neighbourhood
    defined by ``Bc``::

        0 0 0 0 0
        0 0 2 0 0
        0 0 2 0 0
        0 0 3 0 0
        0 0 3 0 0
        0 0 0 0 0

    The top 2 is a local maximum because it has the maximal value in its
    neighbourhood, but it is not a regional maximum.


    Parameters
    ----------
    f : ndarray
    Bc : ndarray, optional
        structuring element
    out : ndarray, optional
        Used for output. Must be Boolean ndarray of same size as `f`
    output : deprecated
        Do not use

    Returns
    -------
    filtered : ndarray
        boolean image of same size as f.

    See Also
    --------
    locmax : function
        Local maxima. The local maxima are a superset of the regional maxima
    '''
    Bc = get_structuring_elem(f, Bc)
    Bc = _remove_centre(Bc.copy())
    output = _get_output(f, out, 'regmax', np.bool_, output=output)
    return _morph.regmin_max(f, Bc, output, False)

def subm(a, b, out=None):
    '''
    c = subm(a, b, out={None})

    Subtract (with saturation).

    This is similar to:

    c = a - b

    but with saturation instead of underflow.

    Examples
    --------

    ::

        a = np.array([10, 10, 10], np.uint8)
        b = np.array([ 5, 10, 15], np.uint8)

        print subm(a,b)

    Prints out::

        [5, 0, 0]

    Parameters
    ----------
    a : ndarray
    b : ndarray
    out : ndarray, optional
        Pass ``a`` as output to subtract in-place.

    Returns
    -------
    c : ndarray
        Result of subtraction
    '''
    if a.dtype != b.dtype:
        raise ValueError('mahotas.subm: This is only well-defined if both arguments are of the same type')
    out = _get_output(a, out, 'subm')
    if out is not a:
        out[:] = a
    return _morph.subm(out, b)


def tophat_close(f, Bc=None, out=None):
    '''
    fclosed = tophat_close(f, Bc={3x3 cross}, out={new array})

    Closed top-hat transform (aka black tophat transform)

    This returns objects that are smaller than ``Bc`` and contain lower values
    than their surroundings.

    See: http://en.wikipedia.org/wiki/Top-hat_transform

    Parameters
    ----------
    f : ndarray
    Bc : ndarray, optional
        structuring element
    out : ndarray, optional
        output array

    Returns
    -------
    fclosed : ndarray
        Of same type and shape as ``f``

    See Also
    --------
    tophat_close : function
        Sister function to this one
    '''
    Bc = get_structuring_elem(f, Bc)
    out = _get_output(f, out, 'tophat_close')
    fc = close(f, Bc)
    return subm(fc, f, out=out)

def tophat_open(f, Bc=None, out=None):
    '''
    fopen = tophat_open(f, Bc={3x3 cross}, out={new array})

    Open top-hat transform (aka white tophat transform)

    This returns objects that are smaller than ``Bc`` and contain higher values
    than their surroundings.

    See: http://en.wikipedia.org/wiki/Top-hat_transform

    Parameters
    ----------
    f : ndarray
    Bc : ndarray, optional
        structuring element
    out : ndarray, optional
        output array

    Returns
    -------
    fopened : ndarray
        Of same type and shape as ``f``

    See Also
    --------
    tophat_close : function
        Sister function to this one
    '''
    Bc = get_structuring_elem(f, Bc)
    out = _get_output(f, out, 'tophat_open')
    fo = open(f,Bc)
    return subm(f, fo, out=out)


def circle_se(radius):
    '''
    circle = circle_se(radius)

    Build a circular structuring element of a given radius

    Parameters
    ----------
    radius : int
        Radius of circle

    Returns
    -------
    circle : boolean ndarray
    '''
    if not (radius > 0):
        raise ValueError('mahotas.morph.circle: radius must be positive')
    X = np.arange(-radius, +radius+1)
    X,Y = np.meshgrid(X,X)
    return (X**2 + Y**2) < radius**2
