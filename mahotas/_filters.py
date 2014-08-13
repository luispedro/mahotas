# Copyright (C) 2010, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

from __future__ import division

mode2int = {
    'nearest' : 0,
    'wrap' : 1,
    'reflect' : 2,
    'mirror' : 3,
    'constant' : 4,
    'ignore' : 5,
}

modes = frozenset(mode2int.keys())

def _checked_mode2int(mode, cval, fname):
    if mode not in modes:
        raise ValueError('mahotas.%s: `mode` not in %s' % (fname, modes))
    if mode == 'constant' and cval != 0.:
        raise NotImplementedError('Please email mahotas developers to get this implemented.')
    return mode2int[mode]

_check_mode = _checked_mode2int

