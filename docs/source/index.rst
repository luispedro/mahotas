==================================
Mahotas: Computer Vision in Python
==================================

.. note:: If you are using mahotas in a scientific publication, please cite:

        Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer
        vision. Journal of Open Research Software 1(1):e3, DOI:
        http://dx.doi.org/10.5334/jors.ac

Mahotas is a computer vision and image processing library for Python.

It includes many algorithms implemented in C++ for speed while operating in
numpy arrays and with a very clean Python interface.

Mahotas currently has over 100 functions for image processing and computer
vision and it keeps growing. Some examples of mahotas functionality:

 - `watershed <http://mahotas.readthedocs.io/en/latest/api.html#mahotas.cwatershed>`__
 - convex points calculations.
 - `hit & miss. thinning <http://mahotas.readthedocs.io/en/latest/api.html#mahotas.hitmiss>`__
 - Zernike & Haralick, `local binary patterns
   <http://mahotas.readthedocs.io/en/latest/lbp.html>`__, and TAS features.
 - `morphological processing <http://mahotas.readthedocs.io/en/latest/morphology.html>`__
 - `Speeded-Up Robust Features (SURF), a form of local features <http://mahotas.readthedocs.io/en/latest/surf.html>`__
 - `thresholding <http://mahotas.readthedocs.io/en/latest/thresholding.html>`__
 - convolution.
 - Sobel edge detection.

The release schedule is roughly one release every few months and each release
brings new functionality and improved performance. The interface is very
stable, though, and code written using a version of mahotas from years back
will work just fine in the current version, except it will be faster (some
interfaces are deprecated and will be removed after a few years, but in the
meanwhile, you only get a warning).

Bug reports with test cases typically get fixed in 24 hours.

.. seealso::

    `mahotas-imread <http://imread.readthedocs.io/en/latest/>`__ is side
    project which includes code to read/write images to files

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
   mahotas-features
   faq
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
