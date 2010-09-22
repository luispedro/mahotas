===================================
Mahotas: Image Processing for numpy
===================================

Contents:

.. toctree::
   :maxdepth: 2

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


Image Processing Library for Python.

It includes a couple of algorithms implemented in C++ for speed while operating
in numpy arrays.

Notable algorithms:
 - watershed.
 - thresholding.
 - convex points calculations.
 - hit & miss. thinning.
 - Zernike & Haralick features.
 - freeimage based numpy image loading (requires freeimage libraries to be
 installed).


Examples
--------

This is a simple example of loading a file (called `test.jpeg`) and calling
`watershed` using above threshold regions as a seed (we use Otsu to define
threshold).

::

    import numpy as np
    from scipy import ndimage
    import mahotas
    import pylab

    img = mahotas.imread('test.jpeg')
    T_otsu = mahotas.thresholding.otsu(img)
    seeds,_ = ndimage.label(img > T_otsu)
    labeled = mahotas.cwatershed(img.max() - img, seeds)

    pylab.imshow(labeled)


Support
-------

*Website*: `http://luispedro.org/software/mahotas
<http://luispedro.org/software/mahotas>`_

*API Docs*: `http://packages.python.org/mahotas/
<http://packages.python.org/mahotas/>`_

*Mailing List*: Use the `pythonvision mailing list
<http://groups.google.com/group/pythonvision?pli=1>`_ for questions, bug
submissions, etc.


.. automodule:: mahotas
    :members: fullhistogram, otsu, rc, stretch, close_holes, get_structuring_elem, dilate, erode, cwatershed, majority_filter, thin, bwperim, center_of_mass, bbox, croptobbox, sobel, euler, moments, distance, imread, imsave, segmentation, features

