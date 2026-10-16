# Copyright (C) 2010-2026, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

import numpy as np
from . import _convex

__all__ = [
    'line',
    'fill_polygon',
    'convexhull',
    'fill_convexhull',
    ]


def line(p0, p1, canvas, color=1):
    '''
    line((y0,x0), (y1,x1), canvas, color=1)

    Draw a line

    Parameters
    ----------
    p0 : pair of integers
        first point
    p1 : pair of integers
        second point
    canvas : ndarray
        where to draw, will be modified in place
    color : integer, optional
        which value to store on the pixels (default: 1)

    notes
    -----
    `Implementation Reference <https://en.wikipedia.org/wiki/Bresenham's_line_algorithm>`__
    '''
    y0,x0 = p0
    y1,x1 = p1
    steep = abs(y1-y0) > abs(x1 -x0)
    if steep:
        x0,y0 = y0,x0
        x1,y1 = y1,x1
    if x0 > x1:
        x0,x1 = x1,x0
        y0,y1 = y1,y0
    dx = x1 - x0
    dy = abs(y1-y0)
    error = dx/2.
    y = y0
    ystep = (+1 if y0 < y1 else -1)
    for x in range(x0,x1+1):
        if steep:
            canvas[x,y] = color
        else:
            canvas[y,x] = color
        error -= dy
        if error < 0:
            y += ystep
            error += dx


def fill_polygon(polygon, canvas, color=1):
    '''
    fill_polygon([(y0,x0), (y1,x1),...], canvas, color=1)

    Draw a filled polygon in canvas

    Parameters
    ----------
    polygon : list of pairs
        a list of (y,x) points with integer coordinates. The polygon is
        closed automatically (the last point connects to the first).
    canvas : ndarray
        where to draw, will be modified in place
    color : integer, optional
        which colour to use (default: 1)

    Notes
    -----
    This is a scanline fill. For each row ``y``, the crossings of the polygon
    edges with that row are computed and the pixels between successive pairs
    of crossings are filled (even-odd rule).

    Edges are half-open in ``y``: an edge from ``y0`` to ``y1`` (``y0 < y1``)
    crosses rows ``y0 < y <= y1``. Therefore, the topmost row of the polygon
    (smallest ``y``) is **not** filled, while the bottom row is. Horizontal
    edges do not fill anything by themselves. Within a row, both ends are
    included: pixel ``x`` is filled if ``floor(left) <= x <= floor(right)``,
    where ``left`` and ``right`` are the crossings.

    For example, the square with corners ``(0,0)`` and ``(4,4)`` fills rows
    1 to 4 (all of columns 0 to 4), but not row 0::

        canvas = np.zeros((5, 5))
        fill_polygon([(0, 0), (0, 4), (4, 4), (4, 0)], canvas)
        # canvas[0] is all zeros; canvas[1:] is all ones

    To include the top row, either start the polygon one row higher (e.g.,
    ``(-1, 0), (-1, 4), (4, 4), (4, 0)``) or also draw the edges with
    ``line``.
    '''
# algorithm adapted from: https://www.alienryderflex.com/polygon_fill/
    if not len(polygon):
        return
    min_y = min(y for y,x in polygon)
    max_y = max(y for y,x in polygon)
    polygon = [(float(y),float(x)) for y,x in polygon]
    # Clip to the canvas: negative indices would otherwise wrap around
    for y in range(max(min_y, 0), min(max_y + 1, canvas.shape[0])):
        nodes = []
        j = -1
        for i,p in enumerate(polygon):
            pj = polygon[j]
            if p[0] < y and pj[0] >= y or pj[0] < y and p[0] >= y:
                dy = pj[0] - p[0]
                if dy:
                    nodes.append( (p[1] + (y-p[0])/(pj[0]-p[0])*(pj[1]-p[1])) )
                elif p[0] == y:
                    nodes.append(p[1])
            j = i
        nodes.sort()
        for n,nn in zip(nodes[::2],nodes[1::2]):
            start = max(int(np.floor(n)), 0)
            stop = int(np.floor(nn)) + 1
            if stop > start:
                canvas[y, start:stop] = color

def convexhull(bwimg):
    '''
    hull = convexhull(bwimg)

    Compute the convex hull as a polygon

    This is an implementation of the Graham Scan:
    https://en.wikipedia.org/wiki/Graham_scan

    Parameters
    ----------
    bwimg : ndarray
        input image (interpreted as boolean). Only 2D arrays are supported.

    Returns
    -------
    hull : ndarray
        Set of (y,x) coordinates of hull corners
    '''
    bwimg = np.ascontiguousarray(bwimg, dtype=np.bool_)
    if bwimg.ndim != 2:
        raise ValueError('mahotas.polygon.convexhull: Only two-dimensional images supported')
    return _convex.convexhull(bwimg)

def fill_convexhull(bwimg):
    '''
    hull = fill_convexhull(bwimg)

    Compute the convex hull and return it as a binary mask

    Parameters
    ----------
    bwimg : ndarray
        input image (interpreted as boolean). Only 2D arrays are supported.

    Returns
    -------
    hull : ndarray
        Image of same shape and dtype as `bwimg` with the hull filled in.
        For boolean input, the hull is ``True``; otherwise, it has value 255.
    '''

    points = convexhull(bwimg)
    canvas = np.zeros_like(bwimg)
    black = (1 if bwimg.dtype == np.bool_ else 255)
    fill_polygon(points, canvas, black)
    canvas[bwimg != 0] = black
    return canvas

