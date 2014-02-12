=======
Mahotas
=======
Python Computer Vision Library
------------------------------

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

Please cite the mahotas paper (see details below under Citation_) if you use
it in a publication.

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

If you compiled from source, **you need to do this in another directory** (or
compile locally, which can be accomplished with ``python setup.py build
--build-lib=.``).

If something fails, you can obtain more detail by running it again in *verbose
mode*::

    python -c "import mahotas; mahotas.test(verbose=True)"

Visual Studio
~~~~~~~~~~~~~

For compiling from source in Visual Studio, use::

    python setup.py build_ext -c msvc
    python setup.py install

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
debug compile. You can set it to the value ``2`` to get extra checks::

    export DEBUG=2
    python setup.py test

Be careful not to use this in production unless you are chasing a bug. The
debug modes are pretty slow as they add many runtime checks.

The ``Makefile`` that is shipped with the source of mahotas can be useful too.
``make debug`` will create a debug build. ``make fast`` will create a non-debug
build (you need to ``make clean`` in between). ``make test`` will run the test
suite.


Travis Build Status
~~~~~~~~~~~~~~~~~~~

.. image:: https://travis-ci.org/luispedro/mahotas.png
       :target: https://travis-ci.org/luispedro/mahotas


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


1.0.4 (2013-12-15)
~~~~~~~~~~~~~~~~~~
- Add mahotas.demos.load()
- Add stretch_rgb() function
- Add demos to mahotas namespace
- Fix SLIC superpixels


1.0.3 (2013-10-06)
~~~~~~~~~~~~~~~~~~
- Add border & as_slice arguments to bbox()
- Better error message in gaussian_filter
- Allow as_rgb() to take integer arguments
- Extend distance() to n-dimensions
- Update to newer Numpy APIs (remove direct access to PyArray members)

1.0.2 (July 10 2013)
~~~~~~~~~~~~~~~~~~~~
- Fix requirements filename

1.0.1 (July 9 2013)
~~~~~~~~~~~~~~~~~~~
- Add lbp_transform() function
- Add rgb2sepia function
- Add mahotas.demos.nuclear_image() function
- Work around matplotlib.imsave's implementation of greyscale
- Fix Haralick bug (report & patch by Tony S Yu)
- Add count_binary1s() function

1.0 (May 21 2013)
~~~~~~~~~~~~~~~~~
- Make matplotlib a soft dependency
- Add demos.image_path() function
- Add citation() function
- Fix a few corner cases in texture analysis
- Integrate with travis
- Update citation (include DOI)

See the `ChangeLog
<https://github.com/luispedro/mahotas/blob/master/ChangeLog>`__ for older
version.
