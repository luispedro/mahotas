# Copyright (C) 2012-2019, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

from __future__ import division

import numpy as np
from .internal import _check_3

def rgb2grey(array, dtype=np.float):
    '''
    grey = rgb2grey(rgb_image, dtype=np.float)

    Convert an RGB image to a grayscale image

    The interpretation of RGB and greyscale values is very much object
    dependent (as anyone who has used an overhead projector which mangled their
    colour figures will have experienced). This function uses a typical method
    for conversion and will work acceptably well for typical use cases, but if
    you have strict requirements, consider implementing the conversion by
    yourself for fine control.

    Parameters
    ----------
    array : ndarray of shape (a,b,3)
    dtype : dtype, optional
        dtype of return

    Returns
    -------
    grey : ndarray of ``dtype``
    '''
    _check_3(array, 'rgb2grey')
    transform = np.array([0.30, 0.59, 0.11])
    transformed = np.dot(array, transform)
    return transformed.astype(dtype, copy=False)
    
rgb2gray = rgb2grey

def _convert(array, matrix, dtype, funcname):
    _check_3(array, funcname)
    h,w,d = array.shape
    array = array.transpose((2,0,1))
    array = array.reshape((3,h*w))
    array = np.dot(matrix, array)
    array = array.reshape((3,h,w))
    array = array.transpose((1,2,0))
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array

def rgb2xyz(rgb, dtype=None):
    '''
    xyz = rgb2xyz(rgb, dtype={float})

    Convert RGB to XYZ coordinates

    The input is interpreted as sRGB. See Wikipedia for more details:

    http://en.wikipedia.org/wiki/SRGB

    Parameters
    ----------
    rgb : ndarray
    dtype : dtype, optional 
        What dtype to return

    Returns
    -------
    xyz : ndarray

    See Also
    --------
    xyz2rgb : function
        The reverse function
    '''
    transformation = np.array([
                [0.4124, 0.3576, 0.1805],
                [0.2126, 0.7152, 0.0722],
                [0.0193, 0.1192, 0.9505],
                ])
    rgb = rgb/255.
    a = 0.055
    rgb_linear_high = np.power( (rgb + a)/(1.+a), 2.4 )
    rgb_linear_low = rgb/12.92
    rgb_linear = np.choose(rgb <= 0.04045, [rgb_linear_low, rgb_linear_high])
    return _convert(rgb_linear, transformation, dtype, 'rgb2xyz')

def xyz2rgb(xyz, dtype=None):
    '''
    rgb = xyz2rgb(xyz, dtype={float})

    Convert XYZ to sRGB coordinates

    The output should be interpreted as sRGB. See Wikipedia for more details:

    http://en.wikipedia.org/wiki/SRGB

    Parameters
    ----------
    xyz : ndarray
    dtype : dtype, optional 
        What dtype to return. Default will be floats

    Returns
    -------
    rgb : ndarray

    See Also
    --------
    rgb2xyz : function
        The reverse function
    '''
    transformation = np.array([
                [ 3.2406, -1.5372, -0.4986],
                [-0.9689,  1.8758,  0.0415],
                [ 0.0557, -0.2040,  1.0570],
                ])
    rgb_linear = _convert(xyz, transformation, dtype, 'xyz2rgb')
    a = 0.055
    srgb_high = (1 + a)*np.power(rgb_linear, 1./2.4)
    srgb_high -= a
    srgb_low = 12.92 * rgb_linear
    srgb = np.choose(rgb_linear <= 0.0031308, [srgb_low, srgb_high])
    srgb *= 255.
    return srgb

def xyz2lab(xyz, dtype=None):
    '''
    lab = xyz2lab(xyz, dtype={float})

    Convert CIE XYZ to L*a*b* coordinates

    http://en.wikipedia.org/wiki/CIELAB

    Parameters
    ----------
    xyz : ndarray
    dtype : dtype, optional 
        What dtype to return. Default will be floats

    Returns
    -------
    lab : ndarray
    '''
    _check_3(xyz, 'xyz2lab')
    x,y,z = xyz.transpose((2,0,1))
    def f(t):
        branch_large = t**(1./3)
        branch_small = ((1/3.)*(29./6)*(29./6))*t + 4/29.
        return np.choose(t <= (6./29)**2, [branch_small, branch_large])
    xn, yn, zn = 0.95047, 1., 1.08883
    fx = f(x/xn)
    fy = f(y/yn)
    fz = f(z/zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    Lab = np.dstack( [L,a,b] )
    if dtype is not None:
        Lab = Lab.astype(dtype, copy=False)
    return Lab

def rgb2lab(rgb, dtype=None):
    '''
    lab = rgb2lab(rgb, dtype={float})

    Convert sRGB to L*a*b* coordinates

    http://en.wikipedia.org/wiki/CIELAB

    Parameters
    ----------
    rgb : ndarray
        Must be of shape (h,w,3)
    dtype : dtype, optional 
        What dtype to return. Default will be floats

    Returns
    -------
    lab : ndarray
    '''
    return xyz2lab(rgb2xyz(rgb), dtype=dtype)

def rgb2sepia(rgb):
    '''
    sepia = rgb2sepia(rgb)

    Parameters
    ----------
    rgb : ndarray
        Must be of shape (h,w,3)

    Returns
    -------
    sepia : ndarray
        Output is of same shape as ``rgb``
    '''
    rgb2sepia_weights = np.array([
                [.393,.769,.189],
                [.349,.686,.168],
                [.272,.534,.131]])
    sepia = _convert(rgb, rgb2sepia_weights, dtype=np.float32, funcname='rgb2sepia')
    sepia = np.minimum(sepia,255)
    sepia = np.maximum(sepia,0)
    return sepia.astype(np.uint8)

