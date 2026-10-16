# Copyright (C) 2010-2026, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

from ._typing import BorderMode

mode2int: dict[BorderMode, int] = {
    'nearest' : 0,
    'wrap' : 1,
    'reflect' : 2,
    'mirror' : 3,
    'constant' : 4,
    'ignore' : 5,
}

modes = frozenset(mode2int.keys())

def _checked_mode2int(mode: BorderMode, cval: float, fname: str) -> int:
    if mode not in modes:
        raise ValueError(f'mahotas.{fname}: `mode` not in {modes}')
    if mode == 'constant' and cval != 0.:
        raise NotImplementedError('Please email mahotas developers to get this implemented.')
    return mode2int[mode]

_check_mode = _checked_mode2int
