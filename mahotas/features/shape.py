# -*- coding: utf-8 -*-
# Copyright (C) 2012-2026, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
import numpy as np
from numpy.typing import ArrayLike
import mahotas as mh
from ..labeled import bwperim
from ..internal import _make_binary

__all__ = [
    'roundness',
    'eccentricity',
    'ellipse_axes',
    ]

def roundness(bw: ArrayLike) -> float | np.float64:
    '''
    r = roundness(bw)

    Roundness

    This is computed as ``P**2 / (4 * pi * A)``, where ``A`` is the area
    (number of pixels) and ``P`` is the number of pixels on the perimeter
    (as computed by ``mahotas.bwperim``). Note that, as ``P`` is a pixel
    count and not a perimeter length, a digital disk gives a value somewhat
    below 1 (around 0.8). Less round shapes give larger values.

    Parameters
    ----------
    bw : ndarray
        Interpreted as a boolean image

    Returns
    -------
    r : float
        Roundness (0 if the image is empty)

    See Also
    --------
    eccentricity : function
        Another shape measure
    '''
    bw = _make_binary(bw)
    area = np.sum(bw)
    perim = np.sum(bwperim(bw))
    if area == 0:
        return 0.
    return float(perim)*perim/4./np.pi/area


def ellipse_axes(bwimage: ArrayLike) -> tuple[float | np.float64, float | np.float64]:
    ''' Parameters of the 'image ellipse'

    semimajor,semiminor = ellipse_axes(bwimage)

    Returns the parameters of the constant intensity ellipse with the same mass
    and second order moments as the original image.

    Parameters
    ----------
    bwimage : ndarray
        Interpreted as a boolean image

    Returns
    -------
    semimajor : float
        Length (in pixels) of the semi-major axis (0 if the image is empty)
    semiminor : float
        Length (in pixels) of the semi-minor axis (0 if the image is empty)

    References
    ----------
    Prokop, RJ, and Reeves, AP.  1992. CVGIP: Graphical Models and Image
    Processing 54(5):438-460

    '''
    from .moments import moments
    bwimage = _make_binary(bwimage)

    if not np.any(bwimage):
        return 0.,0.

    cof = mh.center_of_mass(bwimage)
    hull_mu00 = moments(bwimage, 0, 0, cof)
    hull_mu11 = moments(bwimage, 1, 1, cof)
    hull_mu02 = moments(bwimage, 0, 2, cof)
    hull_mu20 = moments(bwimage, 2, 0, cof)

    semimajor = np.sqrt((2 * (hull_mu20 + hull_mu02 + \
                    np.sqrt((hull_mu20 - hull_mu02)**2 + \
                    4 * hull_mu11**2)))/hull_mu00)

    semiminor = np.sqrt((2 * (hull_mu20 + hull_mu02 - \
                    np.sqrt((hull_mu20 - hull_mu02)**2 + \
                    4 * hull_mu11**2)))/hull_mu00)
    return semimajor, semiminor

def eccentricity(bwimage: ArrayLike) -> float | np.float64:
    """
    ecc = eccentricity(bwimage)

    Compute eccentricity

    This is the eccentricity of the image ellipse (see ``ellipse_axes``),
    i.e., ``sqrt(semimajor**2 - semiminor**2) / semimajor``.

    Parameters
    ----------
    bwimage : ndarray
        Interpreted as a boolean image

    Returns
    -------
    ecc : float
        Eccentricity measure, in the range [0, 1): 0 for a circle (or an
        empty image), closer to 1 for more elongated shapes.

    See Also
    --------
    ellipse_axes : function
        Compute the axes of the image ellipse
    """
    semimajor, semiminor = ellipse_axes(bwimage)
    if semimajor == 0.:
        return 0.
    return  np.sqrt(semimajor**2 - semiminor**2) / semimajor
