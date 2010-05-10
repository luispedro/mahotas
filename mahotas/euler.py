# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: GPL v2, or later

import numpy as np
from scipy import ndimage
_euler_lookup = np.array([
            0,  1,  1,  0,
            1,  0,  2, -1,
            1,  2,  0, -1,
            0, -1, -1,  0,
            ])/4.
_powers = np.array([
    [1, 2],
    [4, 8]
    ])

def euler(f):
    '''
    euler_nr = euler(f)

    Compute the Euler number of image f

    Parameters
    ----------
      f : A 2-D image
    Returns
    -------
      euler_nr : Euler number
    '''
    if f.dtype is not np.bool:
        assert np.all( (f == 0) | (f == 1)), 'mahotas.euler: Non-binary image'
        f = (f != 0)
    value = ndimage.convolve(f.astype(_powers.dtype), _powers)
    return _euler_lookup[value].sum()

