# -*- coding: utf-8 -*-
# Copyright (C) 2012-2013, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
from __future__ import division
import numpy as np
import mahotas as mh
from ..labeled import bwperim
from ..internal import _make_binary

__all__ = [
    'roundness',
    'eccentricity'
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
    bw = _make_binary(bw)
    area = np.sum(bw)
    perim = np.sum(bwperim(bw))
    if area == 0:
        return 0.
    return float(perim)*perim/4./np.pi/area


def eccentricity(bwimage):
    """
    ecc = eccentricity(bwimage)

    Compute eccentricity

    Parameters
    ----------
    bwimage : ndarray
        Interpreted as a boolean image

    Returns
    -------
    r : float
        Eccentricity measure
    """
    from .moments import moments
    bwimage = _make_binary(bwimage)

    if not np.any(bwimage):
        return 0

    cof = mh.center_of_mass(bwimage)
    hull_mu00 = moments(bwimage, 0, 0, cof)
    hull_mu11 = moments(bwimage, 1, 1, cof)
    hull_mu02 = moments(bwimage, 0, 2, cof)
    hull_mu20 = moments(bwimage, 2, 0, cof)

    # Parameters of the 'image ellipse'
    #   (the constant intensity ellipse with the same mass and
    #   second order moments as the original image.)
    #   From Prokop, RJ, and Reeves, AP.  1992. CVGIP: Graphical
    #   Models and Image Processing 54(5):438-460
    semimajor = np.sqrt((2 * (hull_mu20 + hull_mu02 + \
                    np.sqrt((hull_mu20 - hull_mu02)**2 + \
                    4 * hull_mu11**2)))/hull_mu00)

    semiminor = np.sqrt((2 * (hull_mu20 + hull_mu02 - \
                    np.sqrt((hull_mu20 - hull_mu02)**2 + \
                    4 * hull_mu11**2)))/hull_mu00)

    if semimajor == 0.:
        return 0.
    return  np.sqrt(semimajor**2 - semiminor**2) / semimajor

