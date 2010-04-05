# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010 Murphy Lab
# Carnegie Mellon University
# 
# Written by Luis Pedro Coelho <lpc@cmu.edu>
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
#
# For additional information visit http://murphylab.web.cmu.edu or
# send email to murphy@cmu.edu

from __future__ import division
from scipy import ndimage
import nucleidetection

__all__ = [
    'gvoronoi',
    ]

def gvoronoi(labeled):
    r"""
    segmented = gvoronoi(labeled)

    Generalised Voronoi Transform.

    The generalised Voronoi diagram assigns to the pixel (i,j) the label of the nearest
    object (i.e., the value of the nearest non-zero pixel in labeled).

    Inputs
    ------
        * labeled: an array, of a form similar to the return of scipy.ndimage.label()

    Output
    ------
    segmented is of the same size and type as labeled and
        segmented[y,x] is the label of the object at position y,x
    """
    L1,L2 = ndimage.distance_transform_edt(labeled== 0, return_distances=False, return_indices=True)
    return labeled[L1,L2]

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
