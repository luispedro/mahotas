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

The release schedule is roughly one release a month and each release brings new
functionality and improved performance. The interface is very stable, though,
and code written using a version of mahotas from years back will work just fine
in the current version, except it will be faster (some interfaces are
deprecated and will be removed after a few years, but in the meanwhile, you
only get a warning). In a few unfortunate cases, there was a bug in the old
code and your results will change for the better.

Please cite the mahotas paper (see details below under *Citation*) if you use
it in a publication.

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

If something fails, you can obtain more detail by running it again in *verbose
mode*::

    python -c "import mahotas; mahotas.test(verbose=True)"

Development
-----------

Development happens on github (`http://github.com/luispedro/mahotas <https://github.com/luispedro/mahotas>`__).

You can set the ``DEBUG`` environment variable before compilation to get a
debug compile. You can set it to the value ``2`` to get extra checks::

    export DEBUG=2
    python setup.py test

Be careful not to use this in production unless you are chasing a bug. The
debug modes are pretty slow as they add many runtime checks.


Travis Build Status
~~~~~~~~~~~~~~~~~~~

.. image:: https://travis-ci.org/luispedro/mahotas.png
       :target: https://travis-ci.org/luispedro/mahotas

Citation
--------

If you use mahotas on a published publication, please cite:

    **Luis Pedro Coelho** Mahotas: Open source software for scriptable computer
    vision in Journal of Open Research Software, 2013 (in press).


In Bibtex format::

    @article{mahotas,
        author = {Luis Pedro Coelho},
        title = {Mahotas: Open source software for scriptable computer vision},
        journal = {Journal of Open Research Software},
        year = {2013},
        note = {in press},
        volume = {1}
    }


You can access this information using the ``mahotas.citation()`` function.

Contacts
--------

For bug reports and fixes, feel free to use my email: luis@luispedro.org

For more general with achieving certain tasks in Python, the `pythonvision
mailing list <http://groups.google.com/group/pythonvision?pli=1>`__ is a much
better venue and generates a public discussion log for others in the future.
You can use it for mahotas or general computer vision in Python questions.

Recent Changes
--------------

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

0.99 (May 4 2013)
~~~~~~~~~~~~~~~~~
- Make matplotlib a soft dependency
- Add demos.image_path() function
- Add citation() function

This version is **1.0 beta**.

0.9.8 (April 22 2013)
~~~~~~~~~~~~~~~~~~~~~
- Use matplotlib as IO backend (fallback only)
- Compute dense SURF features
- Fix sobel edge filtering (post-processing)
- Faster 1D convultions (including faster Gaussian filtering)
- Location independent tests (run mahotas.tests.run() anywhere)
- Add labeled.is_same_labeling function
- Post filter SLIC for smoother regions
- Fix compilation warnings on several platforms


0.9.7 (February 03 2013)
~~~~~~~~~~~~~~~~~~~~~~~~
- Add ``haralick_features`` function
- Add ``out`` parameter to morph functions which were missing it
- Fix erode() & dilate() with empty structuring elements
- Special case binary erosion/dilation in C-Arrays
- Fix long-standing warning in TAS on zero inputs
- Add ``verbose`` argument to tests.run()
- Add ``circle_se`` to ``morph``
- Allow ``loc(max|min)`` to take floating point inputs
- Add Bernsen local thresholding (``bernsen`` and ``gbernsen`` functions)


See the ``ChangeLog`` for older version.

*Website*: `http://luispedro.org/software/mahotas
<http://luispedro.org/software/mahotas>`_

*API Docs*: `http://mahotas.readthedocs.org/ <http://mahotas.readthedocs.org/>`__

*Mailing List*: Use the `pythonvision mailing list
<http://groups.google.com/group/pythonvision?pli=1>`_ for questions, bug
submissions, etc.

*Author*: Luis Pedro Coelho (with code by Zachary Pincus [from scikits.image],
Peter J. Verveer [from scipy.ndimage], and Davis King [from dlib])

