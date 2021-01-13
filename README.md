# Mahotas

## Python Computer Vision Library

Mahotas is a library of fast computer vision algorithms (all implemented
in C++ for speed) operating over numpy arrays.

![GH Actions Status](https://github.com/luispedro/mahotas/workflows/Python%20Package%20using%20Conda/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/luispedro/mahotas/badge.svg?branch=master)](https://coveralls.io/github/luispedro/mahotas?branch=master)
[![Downloads](https://pepy.tech/badge/mahotas/month)](https://pepy.tech/project/mahotas/month)
[![License](http://badge.kloud51.com/pypi/l/mahotas.svg)](http://opensource.org/licenses/MIT)
[![Install with Anaconda](https://anaconda.org/conda-forge/mahotas/badges/installer/conda.svg)](https://anaconda.org/conda-forge/mahotas)
[![Join the chat at https://gitter.im/luispedro/mahotas](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/luispedro/mahotas)

Python versions 2.7, 3.4+, are supported.

Notable algorithms:

- [watershed](http://mahotas.readthedocs.io/en/latest/distance.html)
- [convex points calculations](http://mahotas.readthedocs.io/en/latest/polygon.html).
- hit & miss, thinning.
- Zernike & Haralick, LBP, and TAS features.
- [Speeded-Up Robust Features
  (SURF)](http://mahotas.readthedocs.io/en/latest/surf.html), a form of local
  features.
- [thresholding](http://mahotas.readthedocs.io/en/latest/thresholding.html).
- convolution.
- Sobel edge detection.
- spline interpolation
- SLIC super pixels.

Mahotas currently has over 100 functions for image processing and
computer vision and it keeps growing.

The release schedule is roughly one release a month and each release
brings new functionality and improved performance. The interface is very
stable, though, and code written using a version of mahotas from years
back will work just fine in the current version, except it will be
faster (some interfaces are deprecated and will be removed after a few
years, but in the meanwhile, you only get a warning). In a few
unfortunate cases, there was a bug in the old code and your results will
change for the better.

Please cite [the mahotas paper](http://dx.doi.org/10.5334/jors.ac) (see
details below under [Citation](#Citation)) if you use it in a publication.

## Examples

This is a simple example (using an example file that is shipped with
mahotas) of calling watershed using above threshold regions as a seed
(we use Otsu to define threshold).

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

Here is a very simple example of using `mahotas.distance` (which
computes a distance map):

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

(This is under [mahotas/demos/distance.py](https://github.com/luispedro/mahotas/blob/master/mahotas/demos/distance.py).)

How to invoke thresholding functions:

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

## Install

If you are using [conda](http://anaconda.org/), you can install mahotas from
[conda-forge](https://conda-forge.github.io/) using the following commands:

    conda config --add channels conda-forge
    conda install mahotas

### Compilation from source

You will need python (naturally), numpy, and a C++ compiler. Then you
should be able to use:

    pip install mahotas

You can test your installation by running:

    python -c "import mahotas as mh; mh.test()"

If you run into issues, the manual has more [extensive documentation on
mahotas
installation](https://mahotas.readthedocs.io/en/latest/install.html),
including how to find pre-built for several platforms.

## Citation

If you use mahotas on a published publication, please cite:

> **Luis Pedro Coelho** Mahotas: Open source software for scriptable
> computer vision in Journal of Open Research Software, vol 1, 2013.
> [[DOI](http://dx.doi.org/10.5334/jors.ac)]

In Bibtex format:

>   @article{mahotas,
>       author = {Luis Pedro Coelho},
>       title = {Mahotas: Open source software for scriptable computer vision},
>       journal = {Journal of Open Research Software},
>       year = {2013},
>       doi = {http://dx.doi.org/10.5334/jors.ac},
>       month = {July},
>       volume = {1}
>   }

You can access this information using the `mahotas.citation()` function.

## Development

Development happens on github
([http://github.com/luispedro/mahotas](https://github.com/luispedro/mahotas)).

You can set the `DEBUG` environment variable before compilation to get a
debug version:

    export DEBUG=1
    python setup.py test

You can set it to the value `2` to get extra checks:

    export DEBUG=2
    python setup.py test

Be careful not to use this in production unless you are chasing a bug.
Debug level 2 is very slow as it adds many runtime checks.

The `Makefile` that is shipped with the source of mahotas can be useful
too. `make debug` will create a debug build. `make fast` will create a
non-debug build (you need to `make clean` in between). `make test` will
run the test suite.

## Links & Contacts

*Documentation*:
[https://mahotas.readthedocs.io/](https://mahotas.readthedocs.io/)

*Issue Tracker*: [github mahotas
issues](https://github.com/luispedro/mahotas/issues)

*Mailing List*: Use the [pythonvision mailing
list](http://groups.google.com/group/pythonvision?pli=1) for questions,
bug submissions, etc. Or ask on [stackoverflow (tag
mahotas)](http://stackoverflow.com/questions/tagged/mahotas)

*Main Author & Maintainer*: [Luis Pedro Coelho](http://luispedro.org)
(follow on [twitter](https://twitter.com/luispedrocoelho) or
[github](https://github.com/luispedro)).

Mahotas also includes code by Zachary Pincus [from scikits.image], Peter
J. Verveer [from scipy.ndimage], and Davis King [from dlib], Christoph
Gohlke, as well as
[others](https://github.com/luispedro/mahotas/graphs/contributors).

[Presentation about mahotas for bioimage
informatics](http://luispedro.org/files/talks/2013/EuBIAS/mahotas.html)

For more general discussion of computer vision in Python, the
[pythonvision mailing
list](http://groups.google.com/group/pythonvision?pli=1) is a much
better venue and generates a public discussion log for others in the
future. You can use it for mahotas or general computer vision in Python
questions.

## Recent Changes

### Version 1.4.11 (Aug 16 2020)

- Convert tests to pytest
- Fix testing for PyPy

### Version 1.4.10 (Jun 11 2020)

- Build wheels automatically (PR #114 by [nathanhillyer](https://github.com/nathanhillyer))

### Version 1.4.9 (Nov 12 2019)

- Fix FreeImage detection (issue #108)

### Version 1.4.8 (Oct 11 2019)

- Fix co-occurrence matrix computation (patch by @databaaz)

### Version 1.4.7 (Jul 10 2019)

- Fix compilation on Windows

### Version 1.4.6 (Jul 10 2019)

- Make watershed work for >2³¹ voxels (issue #102)
- Remove milk from demos
- Improve performance by avoid unnecessary array copies in `cwatershed()`,
  `majority_filter()`, and color conversions
- Fix bug in interpolation

### Version 1.4.5 (Oct 20 2018)
- Upgrade code to newer NumPy API (issue #95)

### Version 1.4.4 (Nov 5 2017)
- Fix bug in Bernsen thresholding (issue #84)

### Version 1.4.3 (Oct 3 2016)
- Fix distribution (add missing `README.md` file)

### Version 1.4.2 (Oct 2 2016)

- Fix `resize\_to` return exactly the requested size
- Fix hard crash when computing texture on arrays with negative values (issue #72)
- Added `distance` argument to haralick features (pull request #76, by
  Guillaume Lemaitre)

### Version 1.4.1 (Dec 20 2015)

-   Add `filter\_labeled` function
-   Fix tests on 32 bit platforms and older versions of numpy

### Version 1.4.0 (July 8 2015)

-   Added `mahotas-features.py` script
-   Add short argument to citation() function
-   Add max\_iter argument to thin() function
-   Fixed labeled.bbox when there is no background (issue \#61, reported
    by Daniel Haehn)
-   bbox now allows dimensions greater than 2 (including when using the
    `as_slice` and `border` arguments)
-   Extended croptobbox for dimensions greater than 2
-   Added use\_x\_minus\_y\_variance option to haralick features
-   Add function `lbp_names`

### Version 1.3.0 (April 28 2015)

-   Improve memory handling in freeimage.write\_multipage
-   Fix moments parameter swap
-   Add labeled.bbox function
-   Add return\_mean and return\_mean\_ptp arguments to haralick
    function
-   Add difference of Gaussians filter (by Jianyu Wang)
-   Add Laplacian filter (by Jianyu Wang)
-   Fix crash in median\_filter when mismatched arguments are passed
-   Fix gaussian\_filter1d for ndim \> 2

### Version 1.2.4 (December 23 2014)

-   Add PIL based IO

### Version 1.2.3 (November 8 2014)

-   Export mean\_filter at top level
-   Fix to Zernike moments computation (reported by Sergey Demurin)
-   Fix compilation in platforms without npy\_float128 (patch by Gabi
    Davar)

### Version 1.2.2 (October 19 2014)

-   Add minlength argument to labeled\_sum
-   Generalize regmax/regmin to work with floating point images
-   Allow floating point inputs to `cwatershed()`
-   Correctly check for float16 & float128 inputs
-   Make sobel into a pure function (i.e., do not normalize its input)
-   Fix sobel filtering

### Version 1.2.1 (July 21 2014)

-   Explicitly set numpy.include\_dirs() in setup.py [patch by Andrew
    Stromnov]

### Version 1.2 (July 17 2014)

-   Export locmax|locmin at the mahotas namespace level
-   Break away ellipse\_axes from eccentricity code as it can be useful
    on its own
-   Add `find()` function
-   Add `mean_filter()` function
-   Fix `cwatershed()` overflow possibility
-   Make labeled functions more flexible in accepting more types
-   Fix crash in `close_holes()` with nD images (for n \> 2)
-   Remove matplotlibwrap
-   Use standard setuptools for building (instead of numpy.distutils)
-   Add `overlay()` function

### Version 1.1.1 (July 4 2014)

-   Fix crash in close\_holes() with nD images (for n \> 2)

### 1.1.0 (February 12 2014)

-   Better error checking
-   Fix interpolation of integer images using order 1
-   Add resize\_to & resize\_rgb\_to
-   Add coveralls coverage
-   Fix SLIC superpixels connectivity
-   Add remove\_regions\_where function
-   Fix hard crash in convolution
-   Fix axis handling in convolve1d
-   Add normalization to moments calculation

See the
[ChangeLog](https://github.com/luispedro/mahotas/blob/master/ChangeLog)
for older version.


## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fluispedro%2Fmahotas.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fluispedro%2Fmahotas?ref=badge_large)
