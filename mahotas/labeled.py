# Copyright (C) 2008-2015, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# LICENSE: MIT

from __future__ import division
import numpy as np
from .morph import get_structuring_elem
from . import _labeled
from .internal import _get_output
from ._filters import _checked_mode2int
import mahotas as mh

__all__ = [
    'bbox',
    'borders',
    'border',
    'bwperim',
    'filter_labeled',
    'label',
    'labeled_sum',
    'labeled_max',
    'labeled_size',
    'relabel',
    'is_same_labeling',
    'perimeter',
    'remove_bordering',
    'remove_regions',
    'remove_regions_where',
    ]

def label(array, Bc=None, out=None, output=None):
    '''
    labeled, nr_objects = label(array, Bc={3x3 cross}, output={new array})

    Label the array, which is interpreted as a binary array

    This is also called *connected component labeled*, where the connectivity
    is defined by the structuring element ``Bc``.

    See: http://en.wikipedia.org/wiki/Connected-component_labeling

    Parameters
    ----------
    array : ndarray
        This will be interpreted as binary array
    Bc : ndarray, optional
        This is the structuring element to use
    out : ndarray, optional
        Output array. Must be a C-array, of type np.int32

    Returns
    -------
    labeled : ndarray
        Labeled result
    nr_objects : int
        Number of objects
    '''
    output = _get_output(array, out, 'labeled.label', np.int32, output=output)
    output[:] = (array != 0)
    Bc = get_structuring_elem(output, Bc)
    nr_objects = _labeled.label(output, Bc)
    return output, nr_objects

def relabel(labeled, inplace=False):
    '''
    relabeled, nr_objs = relabel(labeled, inplace=False)

    Relabeling ensures that ``relabeled`` is a labeled image such that every
    label from 1 to ``relabeled.max()`` is used (0 is reserved for the
    background and is passed through).

    Example::

        labeled,n = label(some_binary_map)
        for region in xrange(n):
            if not good_region(labeled, region + 1):
                # This deletes the region:
                labeled[labeled == (region + 1)] = 0
        relabel(labeled, inplace=True)

    Parameters
    ----------
    relabeled : ndarray of int
        A labeled array
    inplace : boolean, optional
        Whether to perform relabeling inplace, erasing the values in
        ``labeled`` (default: False)

    Returns
    -------
    relabeled: ndarray
    nr_objs : int
        Number of objects

    See Also
    --------
    label : function
    '''
    labeled = _as_labeled(labeled, labeled, 'relabel', inplace=inplace)
    n = _labeled.relabel(labeled)
    return labeled, n

def is_same_labeling(labeled0, labeled1):
    '''
    same = is_same_labeling(labeled0, labeled1)

    Checks whether ``labeled0`` and ``labeled1`` represent the same labeling
    (i.e., whether they are the same except for a possible change of label
    values).

    Note that the background (value 0) is treated differently. Namely

    is_same_labeling(a, b) implies np.all( (a == 0) == (b == 0) )

    Parameters
    ----------
    labeled0 : ndarray of int
        A labeled array
    labeled1 : ndarray of int
        A labeled array

    Returns
    -------
    same : bool
        True if the labelings passed as argument are equivalent

    See Also
    --------
    label : function
    relabel : function
    '''
    labeled0 = _convert_labeled(labeled0)
    labeled1 = _convert_labeled(labeled1)
    return _labeled.is_same_labeling(labeled0, labeled1)


def remove_regions(labeled, regions, inplace=False):
    '''
    removed = remove_regions(labeled, regions, inplace=False):

    Removes the regions in ``regions``. If an elementwise ``in`` operator
    existed, this would be equivalent to the following::

        labeled[ labeled element-wise-in regions ] = 0

    This function **does not** relabel its arguments. You can use the
    ``relabel`` function for that::

        removed = relabel(remove_regions(labeled, regions))

    Or, saving one image allocation::

        removed = relabel(remove_regions(labeled, regions), inplace=True)

    This is the same, but reuses the memory in the relabeling operation.

    Parameters
    ----------
    relabeled : ndarray of int
        A labeled array
    regions : sequence of int
        These regions will be removed
    inplace : boolean, optional
        Whether to perform removal inplace, erasing the values in
        ``labeled`` (default: False)

    Returns
    -------
    removed : ndarray

    See Also
    --------
    relabel : function
        After removing unecessary regions, it is often a good idea to relabel
        your label image.
    '''
    labeled = _as_labeled(labeled, labeled, 'remove_regions', inplace=inplace)
    regions = np.asarray(regions, dtype=np.intc)
    regions = np.unique(regions)
    _labeled.remove_regions(labeled, regions)
    return labeled


def remove_regions_where(labeled, conditions, inplace=False):
    '''Remove regions based on a boolean array

    A region is removed if ``conditions[region-id]`` evaluates true.

    This function **does not** relabel its arguments. You can use the
    ``relabel`` function for that::

        removed = relabel(remove_regions_where(labeled, conditions))

    Or, saving one image allocation::

        removed = relabel(remove_regions(labeled, conditions), inplace=True)

    This is the same, but reuses the memory in the relabeling operation.


    See Also
    --------
    remove_regions : function
        Variation of this function which uses integer indexing
    '''
    regions, = np.where(conditions)
    return remove_regions(labeled, regions, inplace=inplace)


def remove_bordering(labeled, rsize=1, out=None, output=None):
    '''
    slabeled = remove_bordering(labeled, rsize=1, out={np.empty_like(im)})

    Remove objects that are touching the border.

    Pass ``labeled`` as ``out`` to achieve in-place operation.

    Parameters
    ----------
    labeled : ndarray
        Labeled array
    rsize : int or tuple, optional
        Minimum distance to the border (in Manhatan distance) to allow an
        object to survive. May be int or tuple with len == labeled.ndim.
    out : ndarray, optional
        If ``im`` is passed as ``out``, then it operates inline.

    Returns
    -------
    slabeled : ndarray
        Subset of ``labeled``
    '''
    im = labeled
    invalid = set()
    index = [slice(None,None,None) for _ in range(im.ndim)]
    if type(rsize) is not tuple:
        rsize = (rsize,)*im.ndim
    for dim in range(im.ndim):
        for bordering in (
                    slice(rsize[dim]),
                    slice(im.shape[dim]-rsize[dim], None)
                        ):
            index[dim] = bordering
            for val in np.unique(im[tuple(index)].ravel()):
                if val != 0:
                    invalid.add(val)
        index[dim] = slice(None,None,None)
    if out is None and output is not None: #pragma: no cover
        import warnings
        warnings.warn('Using deprecated `output` argument in function `%s`. Please use `out` in the future.' % 'remove_bordering', DeprecationWarning)
        out = output
    if out is None:
        out = im.copy()
    elif out is not im:
        out[:] = im
    for val in invalid:
        out *= (im != val)
    return out


def border(labeled, i, j, Bc=None, out=None, always_return=True, output=None):
    '''
    border_img = border(labeled, i, j, Bc={3x3 cross}, out={np.zeros(labeled.shape, bool)}, always_return=True)

    Compute the border region between `i` and `j` regions.

    A pixel is on the border if it has value `i` (or `j`) and a pixel in its
    neighbourhood (defined by `Bc`) has value `j` (or `i`).

    Parameters
    ----------
    labeled : ndarray of integer type
        input labeled array
    i : integer
    j : integer
    Bc : structure element, optional
    out : ndarray of same shape as `labeled`, dtype=bool, optional
        where to store the output. If ``None``, a new array is allocated
    always_return : bool, optional
        if false, then, in the case where there is no pixel on the border,
        returns ``None``. Otherwise (the default), it always returns an array
        even if it is empty.

    Returns
    -------
    border_img : boolean ndarray
        Pixels are True exactly where there is a border between `i` and `j` in `labeled`
    '''
    Bc = get_structuring_elem(labeled, Bc)
    output = _get_output(labeled, out, 'labeled.border', bool, output=output)
    output.fill(False)
    return _labeled.border(labeled, Bc, output, i, j, bool(always_return))

def borders(labeled, Bc=None, out=None, output=None, mode='constant'):
    '''
    border_img = borders(labeled, Bc={3x3 cross}, out={np.zeros(labeled.shape, bool)})

    Compute border pixels

    A pixel is on a border if it has value `i` and a pixel in its neighbourhood
    (defined by `Bc`) has value `j`, with ``i != j``.

    Parameters
    ----------
    labeled : ndarray of integer type
        input labeled array
    Bc : structure element, optional
    out : ndarray of same shape as `labeled`, dtype=bool, optional
        where to store the output. If ``None``, a new array is allocated
    mode : {'reflect', 'nearest', 'wrap', 'mirror', 'constant' [default], 'ignore'}
        How to handle borders

    Returns
    -------
    border_img : boolean ndarray
        Pixels are True exactly where there is a border in `labeled`
    '''
    Bc = get_structuring_elem(labeled, Bc)
    output = _get_output(labeled, out, 'labeled.borders', bool, output=output)
    output.fill(False)
    return _labeled.borders(labeled, Bc, output, _checked_mode2int(mode, 0.0, 'borders'))


def bwperim(bw, n=4, mode="constant"):
    '''
    perim = bwperim(bw, n=4)

    Find the perimeter of objects in binary images.

    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.

    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.

    Parameters
    ----------
    bw : ndarray
        A black-and-white image (any other image will be converted to black & white)
    n : int, optional
        Connectivity. Must be 4 or 8 (default: 4)
    mode : {'reflect', 'nearest', 'wrap', 'mirror', 'constant' [default], 'ignore'}
        How to handle borders

    Returns
    -------
    perim : ndarray
        A boolean image

    See Also
    --------
    borders : function
        This is a more generic function
    '''
    bw = (bw != 0)
    return bw&borders(bw, n, mode=mode)

def _as_labeled(array, labeled, funcname, inplace='unused'):
    '''

    labeled = _as_labeled(array, labeled, funcname, inplace='unused')

    Parameters
    ----------
    array : ndarray
    labeled : ndarray
    funcname : str
    inplace : bool or str, optional
        handles ``inplace`` arguments
    '''
    if inplace == 'unused':
        labeled = np.require(labeled, dtype=np.intc, requirements="CW")
    elif not inplace:
        labeled = np.array(labeled, dtype=np.intc)
    elif labeled.dtype != np.intc or not labeled.flags.carray:
        raise ValueError('mahotas.labeled.%s: labeled must be a C-array of type int' % funcname)

    if array.shape != labeled.shape:
        raise ValueError('mahotas.labeled.%s: `array` is not the same size as `labeled`' % funcname)
    return labeled


def _convert_labeled(labeled):
    return np.require(labeled, dtype=np.intc, requirements="CW")

def labeled_sum(array, labeled, minlength=None):
    '''
    sums = labeled_sum(array, labeled, minlength=None)

    Labeled sum. sum will be an array of size ``labeled.max() + 1``, where
    ``sum[i]`` is equal to ``np.sum(array[labeled == i])``.

    Parameters
    ----------
    array : ndarray of any type
    labeled : int ndarray
        Label map. This is the same type as returned from ``mahotas.label()``
    minlength : int, optional
        Minimum size of return array. If labeled has fewer than ``minlength``
        regions, 0s are added to the result.  (optional)

    Returns
    -------
    sums : 1-d ndarray of ``array.dtype``
    '''
    labeled = _as_labeled(array, labeled, 'labeled_sum')
    maxv = labeled.max() + 1
    if minlength is not None:
        maxv = max(maxv, minlength)
    output = np.empty(maxv, dtype=array.dtype)
    _labeled.labeled_sum(array, labeled, output)
    return output


def labeled_max(array, labeled):
    '''
    mins = labeled_max(array, labeled, minlength=None)

    Labeled minimum. ``mins`` will be an array of size ``labeled.max() + 1``, where
    ``mins[i]`` is equal to ``np.min(array[labeled == i])``.

    Parameters
    ----------
    array : ndarray of any type
    labeled : int ndarray
        Label map. This is the same type as returned from ``mahotas.label()``

    Returns
    -------
    mins : 1-d ndarray of ``array.dtype``
    '''
    labeled = _as_labeled(array, labeled, 'labeled_max')
    maxv = labeled.max() + 1
    output = np.empty(maxv, dtype=array.dtype)
    _labeled.labeled_max_min(array, labeled, output, True)
    return output


def labeled_min(array, labeled):
    '''
    maxs = labeled_min(array, labeled)

    Labeled maximum. ``maxs`` will be an array of size ``labeled.max() + 1``, where
    ``maxs[i]`` is equal to ``np.max(array[labeled == i])``.

    Parameters
    ----------
    array : ndarray of any type
    labeled : int ndarray
        Label map. This is the same type as returned from ``mahotas.label()``

    Returns
    -------
    maxs : 1-d ndarray of ``array.dtype``
    '''
    labeled = _as_labeled(array, labeled, 'labeled_min')
    maxv = labeled.max() + 1
    output = np.empty(maxv, dtype=array.dtype)
    _labeled.labeled_max_min(array, labeled, output, False)
    return output

def labeled_size(labeled):
    '''
    sizes = labeled_size(labeled)

    Equivalent to::

        for i in range(...):
            sizes[i] = np.sum(labeled == i)

    but, naturally, much faster.

    Parameters
    ----------
    labeled : int ndarray

    Returns
    -------
    sizes : 1-d ndarray of int

    See Also
    --------
    mahotas.fullhistogram : almost same function by another name (the only
    difference is that that function only accepts unsigned integer types).
    '''
    from .histogram import fullhistogram
    return fullhistogram(labeled.astype(np.uint32))


_perimeter_magic = np.array([
                [10, 2, 10],
                [ 2, 1,  2],
                [10, 2, 10]], np.uint8)
_perimeter_values = None

# This implementation was adapted from scikit-image's implementation
def perimeter(bwimage, n=4, mode="constant"):
    """
    p = perimeter(bwimage, n=4, mode="constant")

    Calculate total perimeter of all objects in binary image.

    Parameters
    ----------
    bwimage : array
        binary image
    n : int, optional
        passed to ``bwperim`` as is
    mode : str, optional
        passed to ``bwperim`` as is

    Returns
    -------
    p : float
        total perimeter of all objects in binary image

    See Also
    --------
    bwperim : function
        Finds the perimeter region

    References
    ----------
    .. [1] K. Benkrid, D. Crookes. Design and FPGA Implementation of
           a Perimeter Estimator. The Queen's University of Belfast.
           http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc
    """
    global _perimeter_values
    perim = bwperim(bwimage, n, mode)
    perim = perim.astype(np.uint8)

    histogram = mh.fullhistogram(
                    mh.convolve(perim, _perimeter_magic))

    if _perimeter_values is None:
        _perimeter_values = np.zeros(34, float)
        _perimeter_values[[5, 7, 15, 17, 25, 27]] = 1
        _perimeter_values[[21, 33]] = np.sqrt(2)
        _perimeter_values[[13, 23]] = (1 + np.sqrt(2)) / 2

    size = min(34, len(histogram))
    return np.dot(histogram[:size], _perimeter_values[:size])


def bbox(f, as_slice=False):
    '''
    Bounding boxes of all objects in a labeled array.

    After::

        bboxes = mh.labeled.bbox(f)

    ``bboxes[34]`` will contain the bounding box of ``(f == 34)``.

    Parameters
    ----------
    f : integer ndarray
    as_slice : boolean, optional
        Whether to return slice objects instead of integer coordinates
        (default: False).

    Returns
    -------
    bboxes : ndarray

    See Also
    --------
    mh.bbox : the binary version of this function
    '''
    import mahotas._bbox
    n = f.max()
    output = np.empty( f.ndim * 2 * (n+1), np.intp)
    output = mahotas._bbox.bbox_labeled(f, output)
    output = output.reshape((n+1, 2*f.ndim))
    if as_slice:
        output = [tuple([slice(s,e) for s,e in r.reshape((-1,2))]) for r in output]
    return output

def filter_labeled(labeled, remove_bordering=False, min_size=None, max_size=None):
    '''Filter labeled regions based on a series of conditions

    .. versionadded :: 1.4.1

    Parameters
    ----------
    labeled : labeled array
    remove_bordering : bool, optional
        whether to remove regions that touch the border
    min_size : int, optional
        Minimum size (in pixels) of objects to keep (default is no minimum)
    max_size : int, optional
        Maximum size (in pixels) of objects to keep (default is no maximum)

    Returns
    -------
    filtered : labeled array
    nr : int
        number of new labels
    '''
    from mahotas.labeled import remove_regions, labeled_size
    import mahotas as mh
    labeled = _as_labeled(labeled, labeled, 'filter_labeled')
    if remove_bordering:
        labeled = mh.labeled.remove_bordering(labeled)
        labeled,nr = mh.labeled.relabel(labeled)
    else:
        nr = labeled.max()

    to_keep = np.ones(nr+1, bool)

    if min_size is not None or max_size is not None:
        sizes = labeled_size(labeled)
        if min_size:
            to_keep &= (sizes >= min_size)
        if max_size:
            to_keep &= (sizes <= max_size)
    to_keep[0] = True
    to_remove = np.where(~to_keep)
    labeled = remove_regions(labeled, to_remove)
    labeled,nr = mh.labeled.relabel(labeled, inplace=True)
    return labeled, nr
