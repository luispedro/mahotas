# -*- coding: utf-8 -*-
# Copyright (C) 2006-2010  Luis Pedro Coelho <luis@luispedro.org>
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

__all__ = ['thin']

def thin(binimg):
    """
    skel = thin(binimg)

    Skeletonisation by thinning

    Parameters
    ----------
    binimg : Binary input image

    Returns
    -------
    skel : Skeletonised version of `binimg`
    """
    from .bbox import bbox
    from ._thin import thin as _thin

    res = np.zeros_like(binimg)
    min0,max0,min1,max1 = bbox(binimg)
    r,c = (max0-min0,max1-min1)

    image_exp = np.zeros((r+2, c+2), bool)
    image_exp[1:r+1, 1:c+1] = binimg[min0:max0,min1:max1]
    imagebuf = np.empty((r+2,c+2), bool)

    _thin(image_exp, imagebuf)
    res[min0:max0,min1:max1] = image_exp[1:r+1, 1:c+1]
    return res

