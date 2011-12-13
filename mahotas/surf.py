# Copyright (C) 2010, Luis Pedro Coelho <luis@luispedro.org>
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
from . import _surf
from .morph import _verify_is_integer_type

__all__ = ['integral', 'surf']

def integral(f, in_place=False, dtype=np.double):
    '''
    fi = integral(f, in_place=False, dtype=np.double):

    Compute integral image

    Parameters
    ----------
    f : ndarray
        input image. Only 2-D images are supported.
    in_place : bool, optional
        Whether to overwrite `f` (default: False).
    dtype : dtype, optional
        dtype to use (default: double)

    Returns
    -------
    fi : ndarray of `dtype` of same shape as `f`
        The integral image
    '''
    if len(f.shape) != 2:
        raise ValueError('mahotas.surf.integral: Can only handle images with 2D (i.e., greyscale images).')
    if not in_place:
        if dtype != f.dtype:
            f = f.astype(dtype)
        else:
            f = f.copy()
    return _surf.integral(f)

def surf(f, nr_octaves=4, nr_scales=6, initial_step_size=1, threshold=0.1, max_points=1024, descriptor_only=False):
    '''
    points = surf(f, nr_octaves=4, nr_scales=6, initial_step_size=1, threshold=0.1, max_points=1024, descriptor_only=False):

    Run SURF detection and descriptor computations

    Parameters
    ----------
    f : ndarray
        input image
    nr_octaves : integer, optional
        Nr of octaves (default: 4)
    nr_scales : integer, optional
        Nr of scales (default: 6)
    initial_step_size : integer, optional
        Initial step size in pixels (default: 1)
    threshold : float, optional
        Threshold of the strength of the interest point (default: 0.1)
    max_points : integer, optional
        Maximum number of points to return. By default, return at most 1024
        points. Note that the number may be smaller even in the case where
        there are that many points. This is a side-effect of the way the
        threshold is implemented: only ``max_points`` are considered, but some
        of those may be filtered out.
    descriptor_only : boolean, optional
        If ``descriptor_only``, then returns only the 64-element descriptors

    Returns
    -------
    points : ndarray of double, shape = (N, 6 + 64)
        `N` is nr of points. Each point is represented as
        *(y,x,scale,score,laplacian,angle, D_0,...,D_63)* where *y,x,scale* is
        the position, *angle* the orientation, *score* and *laplacian* the
        score and sign of the detector; and *D_i* is the descriptor

        If ``descriptor_only``, then only the *D_i*s are returned and the array
        has shape (N, 64)!
    '''
    surfs = _surf.surf(integral(f), nr_octaves, nr_scales, initial_step_size, threshold, max_points)
    if descriptor_only:
        surfs = surfs[:,6:]
    return surfs


def interest_points(f, nr_octaves=4, nr_scales=6, initial_step_size=1, threshold=0.1, max_points=None, is_integral=False):
    '''
    desc_array = interest_points(f, nr_octaves=4, nr_scales=6, initial_step_size=1, threshold=0.1, max_points={all}, is_integral=False)

    SURF Detector

    Parameters
    ----------
    f : ndarray
        input image or integral image (if `is_integral`)
    nr_octaves : integer, optional
        Nr of octaves (default: 4)
    nr_scales : integer, optional
        Nr of scales (default: 6)
    initial_step_size : integer, optional
        Initial step size in pixels (default: 1)
    threshold : float, optional
        Threshold of the strength of the interest point (default: 0.1)
    max_points : integer, optional
        Maximum number of points to return. By default, return all.
    is_integral : boolean, optional
        Whether `f` is an integral image

    Returns
    -------
    points : ndarray of double, shape = (N, 5)
        `N` is nr of points. Each point is represented as
        *(y,x,scale,score,laplacian)* where *y,x,scale* is
        the position, *score* and *laplacian* the score and sign of the
        detector.

    See Also
    --------
    surf : SURF detection and descriptors
    descriptors : SURF descriptors
    '''
    if not is_integral:
        f = integral(f)
    else:
        if f.dtype != np.double:
            raise TypeError('mahotas.surf: integral image must be of dtype double')
    if max_points is None:
        max_points = -1
    return _surf.interest_points(f, nr_octaves, nr_scales, initial_step_size, threshold, max_points)


def descriptors(f, interest_points, is_integral=False, descriptor_only=False):
    '''
    desc_array = descriptors(f, interest_points, is_integral=False)

    Compute SURF descriptors

    Parameters
    ----------
    f : ndarray
        input image or integral image (if `is_integral`)
    interest_points : ndarray
        interest points in the format returned by the ``interest_points()`` function
    is_integral : boolean, optional
        Whether `f` is an integral image
    descriptor_only : boolean, optional
        If ``descriptor_only``, then returns only the 64-element descriptors

    Returns
    -------
    points : ndarray of double, shape = (N, 6 + 64)
        `N` is nr of points. Each point is represented as
        *(y,x,scale,score,laplacian,angle, D_0,...,D_63)* where *y,x,scale* is
        the position, *angle* the orientation, *score* and *laplacian* the
        score and sign of the detector; and *D_i* is the descriptor.
        If ``descriptor_only`` is true, then returns only *(D_0,...,D_63)*
    '''
    if not is_integral:
        f = integral(f)
    else:
        if f.dtype != np.double:
            raise TypeError('mahotas.surf: integral image must be of dtype double')
    surfs = _surf.descriptors(f, interest_points)
    if descriptor_only:
        surfs = surfs[:,6:]
    return surfs


def show_surf(f, spoints, values=None, colors=None):
    '''
    f2 = show_surf(f, spoints, values=None, colors={[(255,0,0)]}):

    Parameters
    ----------
    f : image
        original image
    spoints : ndarray
        output of `surf`
    values : ndarray, same length as `spoints`, optional
        You can pass classes for each point here. If it is not used, then all
        the points are displayed the same way (or, equivalently,
        ``values=np.zeros(len(spoints))``).
    colors : ndarray, length must be same as ``values.max()``, optional
        points with values ``vi`` will have colour ``colors[vi]``.
    '''
    import mahotas.polygon
    if values is None:
        values = np.zeros(len(spoints), int)
        if colors is None:
            colors = [(255,0,0)]
    if colors is None:
        raise NotImplementedError('mahotas.surf.show_surf: colors is None, but values is not')
    def rotate(y,x, a):
        sa = np.sin(a)
        ca = np.cos(a)
        return (ca*x-sa*y, sa*x+ca*y)

    f2 = np.dstack([f,f,f])

    for p,vi in zip(spoints, values):
        y = p[0]
        x = p[1]
        scale = p[2]
        angle = p[5]
        size = int(scale*10)
        y0 = int(y) - size//2
        x0 = int(x) - size//2
        x1 = x + size
        y1 = y + size
        def rotate_around((p0,p1),(c0,c1), a):
            d0 = p0-c0
            d1 = p1 - c1
            d0,d1 = rotate(d0,d1,a)
            return int(c0+d0), int(c1+d1)
        polygon = [(y0,x0), (y0,x1), (y1,x1), (y1,x0), (y0,x0)]
        polygon = [rotate_around(p, (y,x), angle) for p in polygon]
        for p0,p1 in zip(polygon[:-1], polygon[1:]):
            mahotas.polygon.line(p0,p1, f2, color=colors[vi])
    return f2.astype(np.uint8)
