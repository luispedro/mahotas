# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: GPL v2, or later

import numpy as np
from scipy import ndimage
_euler_lookup4 = np.array([
            0,  1,  1,  0,
            1,  0,  2, -1,
            1,  2,  0, -1,
            0, -1, -1,  0,
            ])/4.
_euler_lookup8 = np.array([
            0,  1,  1,  0,
            1,  0, -2, -1,
            1, -2,  0, -1,
            0, -1, -1,  0,
            ])/4.
_powers = np.array([
    [1, 2],
    [4, 8]
    ])

def euler(f, n=8):
    '''
    euler_nr = euler(f, n=8)

    Compute the Euler number of image f

    The Euler number is also known as the Euler characteristic given that many
    other mathematical objects are also known as Euler numbers.

    Parameters
    ----------
      f : A 2-D binary image
      n : Connectivity, one of (4,8). default: 8
    Returns
    -------
      euler_nr : Euler number

    Reference
    ---------
    http://en.wikipedia.org/wiki/Euler_characteristic

    Reference for Implementation
    ----------------------------

    *A Fast Algorithm for Computing the Euler Number of an Image and its VLSI
    Implementation*, doi: 10.1109/ICVD.2000.812628
    '''
    if n == 8:
        lookup = _euler_lookup8
    elif n == 4:
        lookup = _euler_lookup4
    else:
        raise ValueError('mahotas.euler: Connectivity must be 4 or 8')
    if f.dtype is not np.bool:
        assert np.all( (f == 0) | (f == 1)), 'mahotas.euler: Non-binary image'
        f = (f != 0)
    value = ndimage.convolve(f.astype(_powers.dtype), _powers)
    return lookup[value].sum()

