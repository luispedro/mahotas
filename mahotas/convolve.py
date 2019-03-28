# Copyright (C) 2010-2019, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING file)


import numpy as np
from . import _convolve
from . import morph
from .internal import _get_output, _normalize_sequence, _verify_is_floatingpoint_type, _as_floating_point_array
from ._filters import mode2int, modes, _check_mode

__all__ = [
    'convolve',
    'convolve1d',
    'daubechies',
    'idaubechies',
    'find',
    'haar',
    'ihaar',
    'median_filter',
    'rank_filter',
    'template_match',
    'gaussian_filter1d',
    'gaussian_filter',
    'wavelet_center',
    'wavelet_decenter',
    'laplacian_2D'
    ]

def convolve(f, weights, mode='reflect', cval=0.0, out=None, output=None):
    '''
    convolved = convolve(f, weights, mode='reflect', cval=0.0, out={new array})

    Convolution of `f` and `weights`

    Convolution is performed in `doubles` to avoid over/underflow, but the
    result is then cast to `f.dtype`. **This conversion may result in
    over/underflow when using small integer types or unsigned types (if the
    output is negative).** Converting to a floating point representation avoids
    this issue::

        c = convolve(f.astype(float), kernel)

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    weights : ndarray
        weight filter. If not of the same dtype as `f`, it is cast
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    out : ndarray, optional
        Output array. Must have same shape and dtype as `f` as well as be
        C-contiguous.

    Returns
    -------
    convolved : ndarray of same dtype as `f`
    '''
    weights = weights.astype(f.dtype, copy=False)
    if f.ndim != weights.ndim:
        raise ValueError('mahotas.convolve: `f` and `weights` must have the same dimensions')
    output = _get_output(f, out, 'convolve', output=output)
    _check_mode(mode, cval, 'convolve')
    return _convolve.convolve(f, weights, output, mode2int[mode])


def convolve1d(f, weights, axis, mode='reflect', cval=0., out=None):
    '''
    convolved = convolve1d(f, weights, axis, mode='reflect', cval=0.0, out={new array})

    Convolution of `f` and `weights` along axis `axis`.

    Convolution is performed in `doubles` to avoid over/underflow, but the
    result is then cast to `f.dtype`.

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    weights : 1-D ndarray
        weight filter. If not of the same dtype as `f`, it is cast
    axis : int
        Axis along which to convolve
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    out : ndarray, optional
        Output array. Must have same shape and dtype as `f` as well as be
        C-contiguous.

    Returns
    -------
    convolved : ndarray of same dtype as `f`

    See Also
    --------
    convolve : function
        generic convolution
    '''
    weights = np.asanyarray(weights)
    weights = weights.squeeze()
    if weights.ndim != 1:
        raise ValueError('mahotas.convolve1d: only 1-D sequences allowed')
    _check_mode(mode, cval, 'convolve1d')
    if f.flags.contiguous and len(weights) < f.shape[axis]:
        weights = weights.astype(np.double, copy=False)
        indices = [a for a in range(f.ndim) if a != axis] + [axis]
        rindices = [indices.index(a) for a in range(f.ndim)]
        oshape = f.shape
        f = f.transpose(indices)
        tshape = f.shape
        f = f.reshape((-1, f.shape[-1]))

        out = _get_output(f, out, 'convolve1d')
        _convolve.convolve1d(f, weights, out, mode2int[mode])
        out = out.reshape(tshape)
        out = out.transpose(rindices)
        out = out.reshape(oshape)
        return out
    else:
        index = [None] * f.ndim
        index[axis] = slice(0, None)
        weights = weights[tuple(index)]
        return convolve(f, weights, mode=mode, cval=cval, out=out)


def median_filter(f, Bc=None, mode='reflect', cval=0.0, out=None, output=None):
    '''
    median = median_filter(f, Bc={square}, mode='reflect', cval=0.0, out={np.empty(f.shape, f.dtype})

    Median filter

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    Bc : ndarray or int, optional
        Defines the neighbourhood, default is a square of side 3.
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    out : ndarray, optional
        Output array. Must have same shape and dtype as `f` as well as be
        C-contiguous.

    Returns
    -------
    median : ndarray of same type and shape as ``f``
        median[i,j] is the median value of the points in f close to (i,j)
    '''
    if Bc is None:
        Bc = np.ones((3,) * len(f.shape), f.dtype)
    elif f.dtype != Bc.dtype:
        Bc = Bc.astype(f.dtype)
    if f.ndim != Bc.ndim:
        raise ValueError('mahotas.median_filter: `f` and `Bc` must have the same number of dimensions')
    rank = Bc.sum()//2
    output = _get_output(f, out, 'median_filter', output=output)
    _check_mode(mode, cval, 'median_filter')
    return _convolve.rank_filter(f, Bc, output, int(rank), mode2int[mode])

def mean_filter(f, Bc, mode='ignore', cval=0.0, out=None):
    '''mean = mean_filter(f, Bc, mode='ignore', cval=0.0, out=None)

    Mean filter. The value at ``mean[i,j]`` will be the mean of the values in
    the neighbourhood defined by ``Bc``.

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    Bc : ndarray
        Defines the neighbourhood. Must be explicitly passed, no default.
    mode : {'reflect', 'nearest', 'wrap', 'mirror', 'constant', 'ignore' [ignore]}
        How to handle borders. The default is to ignore points beyond the
        border, so that the means computed near the border include fewer elements.
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    out : ndarray, optional
        Output array. Must be a double array with the same shape as `f` as well
        as be C-contiguous.

    Returns
    -------
    mean : ndarray of type double and same shape as ``f``

    See Also
    --------
    median_filter : An alternative filtering method
    '''
    Bc = morph.get_structuring_elem(f, Bc)
    out = _get_output(f, out, 'mean_filter', dtype=np.float64)
    _check_mode(mode, cval, 'mean_filter')
    return _convolve.mean_filter(f, Bc, out, mode2int[mode], cval)


def rank_filter(f, Bc, rank, mode='reflect', cval=0.0, out=None, output=None):
    '''
    ranked = rank_filter(f, Bc, rank, mode='reflect', cval=0.0, out=None)

    Rank filter. The value at ``ranked[i,j]`` will be the ``rank``\ th largest in
    the neighbourhood defined by ``Bc``.

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    Bc : ndarray
        Defines the neighbourhood. Must be explicitly passed, no default.
    rank : integer
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    out : ndarray, optional
        Output array. Must have same shape and dtype as `f` as well as be
        C-contiguous.

    Returns
    -------
    ranked : ndarray of same type and shape as ``f``
        ranked[i,j] is the ``rank``\ th value of the points in f close to (i,j)

    See Also
    --------
    median_filter : A special case of rank_filter
    '''
    Bc = morph.get_structuring_elem(f, Bc)
    output = _get_output(f, out, 'rank_filter', output=output)
    _check_mode(mode, cval, 'rank_filter')
    return _convolve.rank_filter(f, Bc, output, rank, mode2int[mode])


def template_match(f, template, mode='reflect', cval=0., out=None, output=None):
    '''Match template to image

    match = template_match(f, template, mode='reflect', cval=0., out={np.empty_like(f)})

    The value at ``match[i,j]`` will be the difference (in squared euclidean
    terms), between `template` and a same sized window on `f` centered on that
    point.

    Note that the computation is performed using the same dtype as ``f``. Thus
    is may overflow if the template is large.

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    template : ndarray
        Template to match. Must be explicitly passed, no default.
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    out : ndarray, optional
        Output array. Must have same shape and dtype as `f` as well as be
        C-contiguous.

    Returns
    -------
    match : ndarray of same type and shape as ``f``
        match[i,j] is the squared euclidean distance between
        ``f[i-s0:i+s0,j-s1:j+s1]`` and ``template`` (for appropriately defined
        ``s0`` and ``s1``).
    '''
    template = template.astype(f.dtype, copy=False)
    output = _get_output(f, out, 'template_match', output=output)
    _check_mode(mode, cval, 'template_match')
    return _convolve.template_match(f, template, output, mode2int[mode], 0)

def find(f, template):
    '''Match template to image exactly

    coordinates = find(f, template)

    The output is in the same format as the ``np.where`` function.

    Parameters
    ----------
    f : ndarray
        input. Currently, only 2-dimensional images are supported.
    template : ndarray
        Template to match. Must be explicitly passed, no default.

    Returns
    -------
    match : np.array
    coordinates : np.array
        These are the coordinates of the match. The format is similar to the
        output of ``np.where``, but in an ndarray.

    '''
    if f.ndim != 2:
        raise ValueError('mahotas.find: Cannot handle multi-dimensional images')
    template = template.astype(f.dtype)
    out = np.empty(f.shape, bool)
    return _convolve.find2d(f, template, out)


def gaussian_filter1d(array, sigma, axis=-1, order=0, mode='reflect', cval=0., out=None, output=None):
    """
    filtered = gaussian_filter1d(array, sigma, axis=-1, order=0, mode='reflect', cval=0., out={np.empty_like(array)})

    One-dimensional Gaussian filter.

    Parameters
    ----------
    array : ndarray
        input array of a floating-point type

    sigma : float
        standard deviation for Gaussian kernel (in pixel units)
    axis : int, optional
        axis to operate on
    order : {0, 1, 2, 3}, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. An order of 1, 2, or 3 corresponds to convolution with
        the first, second or third derivatives of a Gaussian. Higher
        order derivatives are not implemented
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    out : ndarray, optional
        Output array. Must have same shape and dtype as `array` as well as be
        C-contiguous.

    Returns
    -------
    filtered : ndarray
        Filtered version of `array`

    """
    _verify_is_floatingpoint_type(array, 'gaussian_filter1d')
    sigma = float(sigma)
    s2 = sigma*sigma
    # make the length of the filter equal to 4 times the standard
    # deviations:
    lw = int(4.0 * sigma + 0.5)
    if lw <= 0:
        raise ValueError('mahotas.gaussian_filter1d: sigma must be greater or equal to 0.125 [1/8]')
    x = np.arange(2*lw+1, dtype=float)
    x -= lw
    weights = np.exp(x*x/(-2.*s2))
    weights /= np.sum(weights)
    # implement first, second and third order derivatives:
    if order == 0:
        pass
    elif order == 1 : # first derivative
        weights *= -x/s2
    elif order == 2: # second derivative
        weights *= (x*x/s2-1.)/s2
    elif order == 3: # third derivative
        weights *= (3.0 - x*x/s2)*x/(s2*s2)
    else:
        raise ValueError('mahotas.convolve.gaussian_filter1d: Order outside 0..3 not implemented')
    return convolve1d(array, weights, axis, mode, cval, out=output)


def gaussian_filter(array, sigma, order=0, mode='reflect', cval=0., out=None, output=None):
    """
    filtered = gaussian_filter(array, sigma, order=0, mode='reflect', cval=0., out={np.empty_like(array)})

    Multi-dimensional Gaussian filter.

    Parameters
    ----------
    array : ndarray
        input array, any dimension is supported. If the array is an integer
        array, it will be converted to a double array.
    sigma : scalar or sequence of scalars
        standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : {0, 1, 2, 3} or sequence from same set, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number.  An order of 0 corresponds
        to convolution with a Gaussian kernel. An order of 1, 2, or 3
        corresponds to convolution with the first, second or third
        derivatives of a Gaussian. Higher order derivatives are not
        implemented
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant', 'ignore'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    out : ndarray, optional
        Output array. Must have same shape as `array` as well as be
        C-contiguous. If `array` is an integer array, this must be a double
        array; otherwise, it must have the same type as `array`.

    Returns
    -------
    filtered : ndarray
        Filtered version of `array`

    Notes
    -----
    The multi-dimensional filter is implemented as a sequence of
    one-dimensional convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.
    """
    array = _as_floating_point_array(array)
    output = _get_output(array, out, 'gaussian_filter', output=output)
    orders = _normalize_sequence(array, order, 'gaussian_filter')
    sigmas = _normalize_sequence(array, sigma, 'gaussian_filter')
    output[...] = array[...]
    noutput = None
    for axis in range(array.ndim):
        sigma = sigmas[axis]
        order = orders[axis]
        noutput = gaussian_filter1d(output, sigma, axis, order, mode, cval, noutput)
        output,noutput = noutput,output
    return output

def _wavelet_array(f, inline, func):
    f = _as_floating_point_array(f)
    if f.ndim != 2:
        raise ValueError('mahotas.convolve.%s: Only works for 2D images' % func)
    if not inline:
        return f.copy()
    return f



def _wavelet_center_compute(oshape, border=0, dtype=None, cval=0.0):
    for c in range(1, 16+border):
        nshape = 2**(np.floor(np.log2(oshape))+c)
        nshape = nshape.astype(int, copy=False)
        delta = nshape - oshape
        delta //= 2
        if np.min(delta) <= border:
            continue
        position = []
        for d,e in zip(delta, oshape):
            position.append( slice(d, d + e) )
        return nshape, position

def wavelet_center(f, border=0, dtype=float, cval=0.0):
    '''
    fc = wavelet_center(f, border=0, dtype=float, cval=0.0)

    ``fc`` is a centered version of ``f`` with a shape that is composed of
    powers of 2.

    Parameters
    ----------
    f : ndarray
        input image
    border : int, optional
        The border to use (default is no border)
    dtype : type, optional
        Type of ``fc``
    cval : float, optional
        Which value to fill the border with (default is 0)

    Returns
    -------
    fc : ndarray

    See Also
    --------
    wavelet_decenter : function
        Reverse function
    '''
    nshape, position = _wavelet_center_compute(f.shape, border)
    nimage = np.zeros(nshape, dtype=dtype)
    nimage += cval
    nimage[position] = f
    return nimage


def wavelet_decenter(w, oshape, border=0):
    '''
    f = wavelet_decenter(w, oshape, border=0)

    Undoes the effect of ``wavelet_center``

    Parameters
    ----------
    w : ndarray
        Wavelet array
    oshape : tuple
        Desired shape
    border : int, optional
        The desired border. This **must** be the same value as was used for
        ``wavelet_center`` 

    Returns
    -------
    f : ndarray
        This will have shape ``oshape``

    See Also
    --------
    wavelet_center : function
        Forward function
    '''
    nshape, position = _wavelet_center_compute(oshape, border)
    return w[position]



def haar(f, preserve_energy=True, inline=False):
    '''
    t = haar(f, preserve_energy=True, inline=False)

    Haar transform

    Parameters
    ----------
    f : 2-D ndarray
        Input image
    preserve_energy : bool, optional
        Whether to normalise the result so that energy is preserved (the
        default).
    inline : bool, optional
        Whether to write the results to the input image. By default, a new
        image is returned. Integer images are always converted to floating
        point and copied.

    See Also
    --------
    ihaar : function
        Reverse Haar transform
    '''
    f = _wavelet_array(f, inline, 'haar')
    _convolve.haar(f)
    _convolve.haar(f.T)
    if preserve_energy:
        f /= 2.0
    return f

_daubechies_codes = [('D%s' % ci) for ci in range(2,21,2)]
def _daubechies_code(c):
    try:
        return _daubechies_codes.index(c)
    except:
        raise ValueError('mahotas.convolve: Known daubechies codes are {0}. You passed in {1}.'.format(_daubechies_codes, c))

def daubechies(f, code, inline=False):
    '''
    filtered = daubechies(f, code, inline=False)

    Daubechies wavelet transform

    This function works best if the image sizes are powers of 2!

    Parameters
    ----------
    f : ndarray
        2-D image
    code : str
        One of 'D2', 'D4', ... 'D20'
    inline : bool, optional
        Whether to write the results to the input image. By default, a new
        image is returned. Integer images are always converted to floating
        point and copied.

    See Also
    --------
    haar : function
        Haar transform (equivalent to D2)
    '''
    f = _wavelet_array(f, inline, 'daubechies')
    code = _daubechies_code(code)
    _convolve.daubechies(f, code)
    _convolve.daubechies(f.T, code)
    return f


def idaubechies(f, code, inline=False):
    '''
    rfiltered = idaubechies(f, code, inline=False)

    Daubechies wavelet inverse transform

    Parameters
    ----------
    f : ndarray
        2-D image
    code : str
        One of 'D2', 'D4', ... 'D20'
    inline : bool, optional
        Whether to write the results to the input image. By default, a new
        image is returned. Integer images are always converted to floating
        point and copied.

    See Also
    --------
    haar : function
        Haar transform (equivalent to D2)
    '''
    f = _wavelet_array(f, inline, 'idaubechies')
    code = _daubechies_code(code)
    _convolve.idaubechies(f.T, code)
    _convolve.idaubechies(f, code)
    return f


def ihaar(f, preserve_energy=True, inline=False):
    '''
    t = ihaar(f, preserve_energy=True, inline=False)

    Reverse Haar transform

    ``ihaar(haar(f))`` is more or less equal to ``f`` (equal, except for
    possible rounding issues).

    Parameters
    ----------
    f : 2-D ndarray
        Input image. If it is an integer image, it is converted to floating
        point (double).
    preserve_energy : bool, optional
        Whether to normalise the result so that energy is preserved (the
        default).
    inline : bool, optional
        Whether to write the results to the input image. By default, a new
        image is returned. Integer images are always converted to floating
        point and copied.

    Returns
    -------
    f : ndarray

    See Also
    --------
    haar : function
        Forward Haar transform
    '''
    f = _wavelet_array(f, inline, 'ihaar')
    _convolve.ihaar(f)
    _convolve.ihaar(f.T)
    if preserve_energy:
        f *= 2.0
    return f

def laplacian_2D(array, alpha = 0.2):
    """
    filtered = laplacian_2D(array, alpha = 0.2)

    2D Laplacian filter.

    Parameters
    ----------
    array : ndarray
        input 2D array. If the array is an integer array, it will be converted 
        to a double array.
    alpha : scalar or sequence of scalars
        controls the shape of Laplacian operator. Must be 0-1. A larger values 
        makes the operator empahsize the diagonal direction.

    Returns
    -------
    filtered : ndarray
        Filtered version of `array`
    """
    array = np.array(array, dtype=np.float)
    if array.ndim != 2:
        raise ValueError('mahotas.laplacian_2D: Only available for 2-dimensional arrays')
        
    alpha = max(0, min(alpha,1));
    ver_hor_weight = (1. - alpha) / (alpha + 1.)
    diag_weight = alpha / (alpha + 1.)
    center = -4. / (alpha + 1.)
    weights = np.array([
    [diag_weight, ver_hor_weight, diag_weight],
    [ver_hor_weight, center, ver_hor_weight],
    [diag_weight, ver_hor_weight, diag_weight]])
    
    output = convolve(array, weights, mode='nearest')
    return output
