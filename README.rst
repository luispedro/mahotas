=======
Mahotas
=======
Python Computer Vision Library
------------------------------


|Travis|_
|Downloads|_
|License|_

.. |Travis| image:: https://api.travis-ci.org/luispedro/mahotas.png
.. |Downloads| image:: https://pypip.in/d/mahotas/badge.png
.. |License| image:: https://pypip.in/license/mahotas/badge.png
.. _Travis: https://travis-ci.org/luispedro/mahotas
.. _Downloads: https://pypi.python.org/pypi/mahotas
.. _License: http://opensource.org/licenses/MIT


Mahotas is a library of fast computer vision algorithms (all implemented in
C++) operates over numpy arrays for convenience.

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

The release schedule is roughly one release a month and each release brings new
functionality and improved performance. The interface is very stable, though,
and code written using a version of mahotas from years back will work just fine
in the current version, except it will be faster (some interfaces are
deprecated and will be removed after a few years, but in the meanwhile, you
only get a warning). In a few unfortunate cases, there was a bug in the old
code and your results will change for the better.

Please cite `the mahotas paper <http://dx.doi.org/10.5334/jors.ac>`__ (see
details below under Citation_) if you use it in a publication.

Examples
--------

This is a simple example of loading a file (called `test.jpeg`) and calling
`watershed` using above threshold regions as a seed (we use Otsu to define
threshold).

::

    # import using ``mh`` abbreviation which is common:
    import mahotas as mh

    # Load one of the demo images
    im = mh.demos.load('nuclear')

    # Automatically compute a threshold
    T_otsu = mh.thresholding.otsu(im)

    # Label the thresholded image (thresholding is done with numpy operations
    seeds,nr_regions = mh.label(im > T_otsu)

    # Call seeded watershed to expand the threshold
    labeled = mh.cwatershed(im.max() - im, seeds)

Here is a very simple example of using ``mahotas.distance`` (which computes a
distance map)::

    import pylab as p
    import numpy as np
    import mahotas as mh

    f = np.ones((256,256), bool)
    f[200:,240:] = False
    f[128:144,32:48] = False
    # f is basically True with the exception of two islands: one in the lower-right
    # corner, another, middle-left

    dmap = mh.distance(f)
    p.imshow(dmap)
    p.show()

(This is under ``mahotas/demos/distance.py``).

How to invoke thresholding functions::

    import mahotas as mh
    import numpy as np
    from pylab import imshow, gray, show, subplot
    from os import path

    # Load photo of mahotas' author in greyscale
    photo = mh.demos.load('luispedro', as_grey=True)

    # Convert to integer values (using numpy operations)
    photo = photo.astype(np.uint8)

    # Compute Otsu threshold
    T_otsu = mh.otsu(photo)
    thresholded_otsu = (photo > T_otsu)

    # Compute Riddler-Calvard threshold
    T_rc = mh.rc(photo)
    thresholded_rc = (photo > T_rc)

    # Now call pylab functions to display the image
    gray()
    subplot(2,1,1)
    imshow(thresholded_otsu)
    subplot(2,1,2)
    imshow(thresholded_rc)
    show()

As you can see, we rely on numpy/matplotlib for many operations.

Install
-------

You will need python (naturally), numpy, and a C++ compiler. Then you should be
able to use::

    pip install mahotas

You can test your instalation by running::

    python -c "import mahotas; mahotas.test()"


If you run into issues, the manual has more `extensive documentation on mahotas
intallation <http://mahotas.readthedocs.org/en/latest/install.html>`__

Citation
--------

.. _Citation:

If you use mahotas on a published publication, please cite:

    **Luis Pedro Coelho** Mahotas: Open source software for scriptable computer
    vision in Journal of Open Research Software, vol 1, 2013. [`DOI
    <http://dx.doi.org/10.5334/jors.ac>`__]


In Bibtex format::

    @article{mahotas,
        author = {Luis Pedro Coelho},
        title = {Mahotas: Open source software for scriptable computer vision},
        journal = {Journal of Open Research Software},
        year = {2013},
        doi = {http://dx.doi.org/10.5334/jors.ac},
        month = {July},
        volume = {1}
    }


You can access this information using the ``mahotas.citation()`` function.

Development
-----------

Development happens on github (`http://github.com/luispedro/mahotas <https://github.com/luispedro/mahotas>`__).

You can set the ``DEBUG`` environment variable before compilation to get a
debug version::

    export DEBUG=1
    python setup.py test

You can set it to the value ``2`` to get extra checks::

    export DEBUG=2
    python setup.py test

Be careful not to use this in production unless you are chasing a bug. Debug
level 2 is very slow as it adds many runtime checks.

The ``Makefile`` that is shipped with the source of mahotas can be useful too.
``make debug`` will create a debug build. ``make fast`` will create a non-debug
build (you need to ``make clean`` in between). ``make test`` will run the test
suite.


Links & Contacts
----------------


*Documentation*: `http://mahotas.readthedocs.org/ <http://mahotas.readthedocs.org/>`__

*Issue Tracker*: `github mahotas issues <https://github.com/luispedro/mahotas/issues>`__

*Mailing List*: Use the `pythonvision mailing list
<http://groups.google.com/group/pythonvision?pli=1>`_ for questions, bug
submissions, etc. Or ask on `stackoverflow (tag mahotas)
<http://stackoverflow.com/questions/tagged/mahotas>`__

*Main Author & Maintainer*: `Luis Pedro Coelho <http://luispedro.org>`__ (follow on `twitter
<https://twitter.com/luispedrocoelho>`__ or `github
<https://github.com/luispedro>`__).

Mahotas also includes code by Zachary Pincus [from scikits.image], Peter J.
Verveer [from scipy.ndimage], and Davis King [from dlib], Christoph Gohlke, as
well as `others <https://github.com/luispedro/mahotas/graphs/contributors>`__.

`Presentation about mahotas for bioimage informatics
<http://luispedro.org/files/talks/2013/EuBIAS/mahotas.html>`__

For more general discussion of computer vision in Python, the `pythonvision
mailing list <http://groups.google.com/group/pythonvision?pli=1>`__ is a much
better venue and generates a public discussion log for others in the future.
You can use it for mahotas or general computer vision in Python questions.

Recent Changes
--------------

Version 1.2.3 (November 8 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Export mean_filter at top level
- Fix to Zernike moments computation (reported by Sergey Demurin)
- Fix compilation in platforms without npy_float128 (patch by Gabi Davar)


Version 1.2.2 (October 19 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Add minlength argument to labeled_sum
- Generalize regmax/regmin to work with floating point images
- Allow floating point inputs to ``cwatershed()``
- Correctly check for float16 & float128 inputs
- Make sobel into a pure function (i.e., do not normalize its input)
- Fix sobel filtering

Version 1.2.1 (July 21 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Explicitly set numpy.include_dirs() in setup.py [patch by Andrew Stromnov]

Version 1.2 (July 17 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Export locmax|locmin at the mahotas namespace level
- Break away ellipse_axes from eccentricity code as it can be useful on
  its own
- Add ``find()`` function
- Add ``mean_filter()`` function
- Fix ``cwatershed()`` overflow possibility
- Make labeled functions more flexible in accepting more types
- Fix crash in ``close_holes()`` with nD images (for n > 2)
- Remove matplotlibwrap
- Use standard setuptools for building (instead of numpy.distutils)
- Add ``overlay()`` function

Version 1.1.1 (July 4 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Fix crash in close_holes() with nD images (for n > 2)

1.1.0 (February 12 2014)
~~~~~~~~~~~~~~~~~~~~~~~~
- Better error checking
- Fix interpolation of integer images using order 1
- Add resize_to & resize_rgb_to
- Add coveralls coverage
- Fix SLIC superpixels connectivity
- Add remove_regions_where function
- Fix hard crash in convolution
- Fix axis handling in convolve1d
- Add normalization to moments calculation

See the `ChangeLog
<https://github.com/luispedro/mahotas/blob/master/ChangeLog>`__ for older
version.
