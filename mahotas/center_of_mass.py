# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
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

def _mean_out(img, axis):
    if len(img.shape) == 2: return img.mean(1-axis)
    if axis == 0:
        return _mean_out(img.mean(1), 0)
    return _mean_out(img.mean(0), axis - 1)

def center_of_mass(img):
    '''
    x0,x1,... = center_of_mass(img)

    Returns the center of mass of img.
    '''
    xs = []
    for axis,si in enumerate(img.shape):
        xs.append(np.mean(_mean_out(img, axis) * np.arange(si)))
    xs = np.array(xs)
    xs /= img.mean()
    return xs

