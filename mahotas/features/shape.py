# -*- coding: utf-8 -*-
# Copyright (C) 2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
from __future__ import division
import numpy as np
from ..labeled import bwperim

__all__ = [
    'roundness',
    ]

def roundness(bw):
    '''
    r = roundness(bw)

    Roundness

    Parameters
    ----------
    bw : ndarray
        Interpreted as a boolean image

    Returns
    -------
    r : float
    '''
    bw = (bw != 0)
    area = np.sum(bw)
    perim = np.sum(bwperim(bw))
    if area == 0:
        return 0.
    return float(perim)*perim/4./np.pi/area


