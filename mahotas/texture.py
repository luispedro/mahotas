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
from . import _texture

def cooccurence(f, direction, output=None):
    '''
    cooccurence_matrix = cooccurence(f, direction, output={new matrix})

    Compute grey-level cooccurence matrix

    Parameters
    ----------
      f : An integer valued 2-D image
      direction : Direction as index into (horizontal [default], diagonal
                  [nw-se], vertical, diagonal [ne-sw]) 
      output : A np.long 2-D array for the result.
    Returns
    -------
      cooccurence_matrix : cooccurence matrix
    '''
    assert direction in (0,1,2,3), 'mahotas.texture.cooccurence: `direction` %s is not in range(4).' % direction
    if output is None:
        mf = f.max()
        output = np.zeros((mf+1, mf+1), np.long)
    else:
        assert np.min(output.shape) >= f.max(), 'mahotas.texture.cooccurence: output is not large enough'
        assert output.dtype is np.long, 'mahotas.texture.cooccurence: output is not of type np.long'
        output[:] = 0.

    if direction == 2:
        f = f.T
    elif direction == 3:
        f = f[:, ::-1]
    diagonal = (direction in (1,3))
    _texture.cooccurence(f, output, diagonal)
    return output
