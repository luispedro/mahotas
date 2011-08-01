# Copyright (C) 2010-2011, Luis Pedro Coelho <luis@luispedro.org>
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
import _convex

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

    Implementation Reference
    ------------------------

    http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
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
    for x in xrange(x0,x1+1):
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
        a list of (y,x) points
    canvas : ndarray
        where to draw, will be modified in place
    color : integer, optional
        which colour to use (default: 1)
    '''
# algorithm adapted from: http://www.alienryderflex.com/polygon_fill/
    if not polygon:
        return
    min_y = min(y for y,x in polygon)
    max_y = max(y for y,x in polygon)
    polygon = [(float(y),float(x)) for y,x in polygon]
    for y in xrange(min_y, max_y+1):
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
            nn += 1
            canvas[y,n:nn] = color

def convexhull(bwimg):
    '''
    hull = convexhull(bwimg)

    Compute the convex hull as a polygon

    Parameters
    ----------
    bwimg : input image (interpreted as boolean)

    Returns
    -------
    hull : Set of (y,x) coordinates of hull corners
    '''
    Y,X = np.where(bwimg)
    P = list(zip(Y,X))
    if len(P) <= 3:
        return P
    return _convex.convexhull(P)

def fill_convexhull(bwimg):
    '''
    hull = fill_convexhull(bwimg)

    Compute the convex hull and return it as a binary mask

    Parameters
    ----------
    bwimage : input image (interpreted as boolean)

    Returns
    -------
    hull : image of same size and dtype as `bwimg` with the hull filled in.
    '''

    points = convexhull(bwimg)
    canvas = np.zeros_like(bwimg)
    black = (1 if bwimg.dtype == np.bool_ else 255)
    fill_polygon(points, canvas, black)
    canvas[bwimg] = black
    return canvas

