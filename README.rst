=======
Mahotas
=======
Python Computer Vision Library
------------------------------

This library of fast computer vision algorithms (all implemented in C++)
operates over numpy arrays for convenience.

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
 - spline interpolation

Mahotas currently has over 100 functions for image processing and computer
vision and it keeps growing.

There is a `manuscript about mahotas <http://arxiv.org/abs/1211.4907>`__, which
will hopefully evolve into a journal publication later.

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

Here is a very simple example of using ``mahotas.distance`` (which computes a
distance map)::

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

(This is under ``mahotas/demos/distance.py``).

How to invoke thresholding functions::

    import mahotas
    import numpy as np
    from pylab import imshow, gray, show, subplot
    from os import path

    photo = mahotas.imread('luispedro.org', as_grey=True)
    photo = photo.astype(np.uint8)

    T_otsu = mahotas.otsu(photo)
    thresholded_otsu = (photo > T_otsu)

    T_rc = mahotas.rc(photo)
    thresholded_rc = (photo > T_rc)


Install
-------

You will need python (naturally), numpy, and a C++ compiler. Then you should be
able to either

Download the source and then run::

    python setup.py install

or use one of::

    pip install mahotas
    easy_install mahotas

You can test your instalation by running::

    python -c "import mahotas; mahotas.test()"

Development
-----------

Development happens on github (`http://github.com/luispedro/mahotas <https://github.com/luispedro/mahotas>`__).

You can set the ``DEBUG`` environment variable before compilation to get a
debug compile. You can set it to the value ``2`` to get extra checks::

    export DEBUG=2
    python setup.py test

Be careful not to use this in production unless you are chasing a bug. The
debug modes are pretty slow as they add many runtime checks.


Contacts
--------

For bugfixes, feel free to use my email: luis@luispedro.org

For more general with achieving certain tasks in Python, the `pythonvision
mailing list <http://groups.google.com/group/pythonvision?pli=1>`__ is a much
better venue and generates a public discussion log for others in the future.

Recent Changes
--------------
0.9.6 (December 02 2012)
~~~~~~~~~~~~~~~~~~~~~~~~
- Fix ``distance()`` of non-boolean images (issue #24 on github)
- Fix encoding issue on PY3 on Mac OS (issue #25 on github)
- Add ``relabel()`` function
- Add ``remove_regions()`` function in labeled module
- Fix ``median_filter()`` on the borders (respect the ``mode`` argument)
- Add ``mahotas.color`` module for conversion between colour spaces
- Add SLIC Superpixels
- Many improvements to the documentation

0.9.5 (November 05 2012)
~~~~~~~~~~~~~~~~~~~~~~~~
- Fix compilation in older G++
- Faster Otsu thresholding
- Python 3 support without 2to3
- Add ``cdilate`` function
- Add ``subm`` function
- Add tophat transforms (functions ``tophat_close`` and ``tophat_open``)
- Add ``mode`` argument to euler() (patch by Karol M. Langner)
- Add ``mode`` argument to bwperim() & borders() (patch by Karol M. Langner)


0.9.4 (October 10 2012)
~~~~~~~~~~~~~~~~~~~~~~~
- Fix compilation on 32-bit machines (Patch by Christoph Gohlke)

0.9.3 (October 9 2012)
~~~~~~~~~~~~~~~~~~~~~~
- Fix interpolation (Report by Christoph Gohlke)
- Fix second interpolation bug (Report and patch by Christoph Gohlke)
- Update tests to newer numpy
- Enhanced debug mode (compile with DEBUG=2 in environment)
- Faster morph.dilate()
- Add labeled.labeled_max & labeled.labeled_min (This also led to a refactoring
  of the labeled_* code)
- Many documentation fixes


0.9.2 (September 1 2012)
~~~~~~~~~~~~~~~~~~~~~~~~
- Fix compilation on Mac OS X 10.8 (reported by Davide Cittaro)
- Freeimage fixes on Windows by Christoph Gohlke
- Slightly faster _filter implementaiton


0.9.1 (August 28 2012)
~~~~~~~~~~~~~~~~~~~~~~

- Python 3 support (you need to use ``2to3``)
- Haar wavelets (forward and inverse transform)
- Daubechies wavelets (forward and inverse transform)
- Corner case fix in Otsu thresholding
- Add soft_threshold function
- Have polygon.convexhull return an ndarray (instead of a list)
- Memory usage improvements in regmin/regmax/close_holes (first reported
  as issue #9 by thanasi)


0.9 (July 16 2012)
~~~~~~~~~~~~~~~~~~
- Auto-convert integer to double on gaussian_filter (previously, integer
  values would result in zero-valued outputs).
- Check for integer types in (reg|loc)(max|min)
- Use name `out` instead of `output` for output arguments. This matches
  Numpy better
- Switched to MIT License

See the ``ChangeLog`` for older version.

*Website*: `http://luispedro.org/software/mahotas
<http://luispedro.org/software/mahotas>`_

*API Docs*: `http://packages.python.org/mahotas/
<http://packages.python.org/mahotas/>`_

*Mailing List*: Use the `pythonvision mailing list
<http://groups.google.com/group/pythonvision?pli=1>`_ for questions, bug
submissions, etc.

*Author*: Luis Pedro Coelho (with code by Zachary Pincus [from scikits.image],
Peter J. Verveer [from scipy.ndimage], and Davis King [from dlib])

