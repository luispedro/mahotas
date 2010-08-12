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
import _center_of_mass

def center_of_mass(img):
    '''
    coords = center_of_mass(img)
    x0,x1,... = center_of_mass(img)

    Returns the center of mass of img.

    Parameters
    ----------
      img : Any ndarray
    Returns
    -------
      coords : A 1-ndarray of coordinates (size = len(img.shape))
    '''
    return _center_of_mass.center_of_mass(img)

