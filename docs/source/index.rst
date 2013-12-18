==================================
Mahotas: Computer Vision in Python
==================================

Mahotas is a computer vision and image processing library for Python.

It includes many algorithms implemented in C++ for speed while operating in
numpy arrays and with a very clean Python interface.

Notable algorithms:
 - watershed.
 - convex points calculations.
 - hit & miss. thinning.
 - Zernike & Haralick, LBP, and TAS features.
 - freeimage based numpy image loading (requires freeimage libraries to be
   installed).
 - Speeded-Up Robust Features (SURF), a form of local features.
 - thresholding.
 - convolution.
 - Sobel edge detection.

Mahotas currently has over 100 functions for image processing and computer
vision and it keeps growing.

The release schedule is roughly one release a month and each release brings new
functionality and improved performance. The interface is very stable, though,
and code written using a version of mahotas from years back will work just fine
in the current version, except it will be faster (some interfaces are
deprecated and will be removed after a few years, but in the meanwhile, you
only get a warning). In a few unfortunate cases, there was a bug in the old
code and your results will change for the better.

If you are using mahotas in a scientific publication, please cite:

    Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer
    vision. Journal of Open Research Software 1(1):e3, DOI:
    http://dx.doi.org/10.5334/jors.ac

Examples
--------

This is a simple example of loading a file (called `test.jpeg`) and calling
`watershed` using above threshold regions as a seed (we use Otsu to define
threshold).

::

    import numpy as np
    import mahotas
    import pylab

    img = mahotas.imread('test.jpeg')
    T_otsu = mahotas.thresholding.otsu(img)
    seeds,_ = mahotas.label(img > T_otsu)
    labeled = mahotas.cwatershed(img.max() - img, seeds)

    pylab.imshow(labeled)


Computing a distance transform is easy too:


.. plot::
    :include-source:

    import pylab as p
    import numpy as np
    import mahotas

    f = np.ones((256,256), bool)
    f[200:,240:] = False
    f[128:144,32:48] = False
    # f is basically True with the exception of two islands: one in the lower-right
    # corner, another, middle-left

    dmap = mahotas.distance(f)
    p.imshow(dmap)
    p.show()


Full Documentation Contents
---------------------------

Jump to detailed `API Documentation <api.html>`__

.. toctree::
   :maxdepth: 2

   install
   wally
   labeled
   thresholding
   wavelets
   distance
   polygon
   features
   lbp
   surf
   surfref
   morphology
   color
   io
   classification
   edf
   faq
   papers
   internals
   principles
   contributing
   tasks
   history
   api



Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
