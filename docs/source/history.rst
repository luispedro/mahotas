=======
History
=======

Version 1.4.11 (Aug 16 2020)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Convert tests to pytest
- Fix testing for PyPy

Version 1.4.10 (Jun 11 2020)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Build wheels automatically (PR #114 by
  `nathanhillyer <https://github.com/nathanhillyer>`__)
- Fix setup.py requirements (issue #86 with fix by
  `ATolkachev <https://github.com/ATolkachev>`__)

Version 1.4.9 (Nov 12 2019)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix FreeImage detection (issue #108)


Version 1.4.8 (Oct 11 2019)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix co-occurrence matrix computation (patch by @databaaz)

Version 1.4.7 (Jul 10 2019)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix compilation on Windows

Version 1.4.6 (Jul 10 2019)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Make watershed work for >2³¹ voxels (issue #102)
- Remove milk from demos
- Improve performance by avoid unnecessary array copies in ``cwatershed()``,
  ``majority_filter()``, and color conversions
- Fix bug in interpolation

Version 1.4.5 (Oct 20 2018)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Upgrade code to newer NumPy API (issue #95)

Version 1.4.4 (Nov 5 2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix bug in Bernsen thresholding (issue #84)

Version 1.4.3 (Oct 3 2016)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix distribution (add missing README.md file)

Version 1.4.2 (Oct 2 2016)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix ``resize\_to`` return exactly the requested size
- Fix hard crash when computing texture on arrays with negative values
  (issue #72)
- Added ``distance`` argument to haralick features (pull request #76, by
  Guillaume Lemaitre)


Version 1.4.1 (Dec 20 2015)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Add filter_labeled function
- Fix tests on 32 bit platforms and older versions of numpy

Version 1.4.0 (July 8 2015)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Added ``mahotas-features.py`` script
- Add `short` argument to citation() function
- Add `max_iter` argument to thin() function
- Fixed labeled.bbox when there is no background (issue #61, reported by Daniel
  Haehn)
- bbox now allows dimensions greater than 2 (including when using the
  ``as_slice`` and ``border`` arguments)
- Extended croptobbox for dimensions greater than 2
- Added use_x_minus_y_variance option to haralick features
- Add function ``lbp_names``


Version 1.3.0 (April 28 2015)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Improve memory handling in freeimage.write_multipage
- Fix moments parameter swap
- Add labeled.bbox function
- Add `return_mean` and `return_mean_ptp` arguments to haralick function
- Add difference of Gaussians filter (by Jianyu Wang)
- Add Laplacian filter (by Jianyu Wang)
- Fix crash in median_filter when mismatched arguments are passed
- Fix gaussian_filter1d for ndim > 2


Version 1.2.4 (December 23 2014)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Add PIL based IO


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

0.8.1 (June 6 2012)
~~~~~~~~~~~~~~~~~~~
- Fix gaussian_filter bug when order argument was used (reported by John Mark
  Agosta)
- Add morph.cerode
- Improve regmax() & regmin(). Rename previous implementations to locmax() &
  locmin()
- Fix erode() on non-contiguous arrays

0.8 (May 7 2012)
~~~~~~~~~~~~~~~~
- Move features to submodule
- Add morph.open function
- Add morph.regmax & morph.regmin functions
- Add morph.close function
- Fix morph.dilate crash

0.7.3 (March 14 2012)
~~~~~~~~~~~~~~~~~~~~~
- Fix installation of test data
- Greyscale erosion & dilation
- Use imread module (if available)
- Add output argument to erode() & dilate()
- Add 14th Haralick feature (patch by MattyG) --- currently off by default
- Improved zernike interface (zernike_moments)
- Add remove_bordering to labeled
- Faster implementation of ``bwperim``
- Add ``roundness`` shape feature



0.7.2 (February 13 2012)
~~~~~~~~~~~~~~~~~~~~~~~~

There were two minor additions:

- Add as_rgb (especially useful for interactive use)
- Add Gaussian filtering (from scipy.ndimage)

And a few bugfixes:

- Fix type bug in 32 bit machines (Bug report by Lech Wiktor Piotrowski)
- Fix convolve1d
- Fix rank_filter


0.7.1 (January 6 2012)
~~~~~~~~~~~~~~~~~~~~~~

The most important change fixed compilation on Mac OS X

Other changes:

- Add convolve1d
- Check that convolution arguments have right dimensions (instead of
  crashing)
- Add descriptor_only argument to surf.descriptors
- Specify all function signatures on freeimage.py


For version **0.7 (Dec 5 2011)**:

The big change was that the *dependency on scipy was removed*. As part of this
process, the interpolate submodule was added. A few important bug fixes as
well.

- Allow specification of centre in Zernike moment computation
- Fix Local Binary Patterns
- Remove dependency on scipy
- Add interpolate module (from scipy.ndimage)
- Add labeled_sum & labeled_sizes
- gvoronoi no longer depends on scipy
- mahotas is importable without scipy
- Fix bugs in 2D TAS (reported by Jenn Bakal)
- Support for 1-bit monochrome image loading with freeimage
- Fix GIL handling on errors (reported by Gareth McCaughan)
- Fix freeimage for 64-bit computers

Version .6.6 (August 8 2011)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- Fix fill_polygon bug (fix by joferkington)
- Fix Haralick feature 6 (fix by Rita Simões)
- Implement ``morph.get_structuring_element`` for ndim > 2. This implies that
  functions such as ``label()`` now also work in multiple dimensions
- Add median filter & ``rank_filter`` functions
- Add template_match function
- Refactor by use of mahotas.internal
- Better error message for when the compiled modules cannot be loaded
- Update contact email. All docs in numpydoc format now.

For version **0.6.5**
~~~~~~~~~~~~~~~~~~~~~

- Add ``max_points`` & ``descriptor_only`` arguments to mahotas.surf
- Fix haralick for 3-D images (bug report by Rita Simões)
- Better error messages
- Fix hit&miss for non-boolean inputs
- Add ``label()`` function

For version **0.6.4**:

- Fix bug in ``cwatershed()`` when using return_lines=1
- Fix bug in ``cwatershed()`` when using equivalent types for image and markers
- Move tests to mahotas.tests and include them in distribution
- Include ChangeLog in distribution
- Fix compilation on the Mac OS
- Fix compilation warnings on gcc

For version **0.6.3**:

- Improve ``mahotas.stretch()`` function
- Fix corner case in surf (when determinant was zero)
- ``threshold`` argument in mahotas.surf
- imreadfromblob() & imsavetoblob() functions
- ``max_points`` argument for mahotas.surf.interest_points()
- Add ``mahotas.labeled.borders`` function

For version **0.6.2**:

Bugfix release:

- Fix memory leak in _surf
- More robust searching for freeimage
- More functions in mahotas.surf() to retrieve intermediate results
- Improve compilation on Windows (patches by Christoph Gohlke)

For version **0.6.1**:

- Release the GIL in morphological functions
- Convolution
- just_filter option in edge.sobel()
- mahotas.labeled functions
- SURF local features

For version **0.6**:

- Improve Local Binary patterns (faster and better interface)
- Much faster erode() (10x faster)
- Faster dilate() (2x faster)
- TAS for 3D images
- Haralick for 3D images
