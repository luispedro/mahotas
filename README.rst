=======
Mahotas
=======
Python Image Processing Toolkit
-------------------------------

Image Processing Library for Python.

It includes a couple of algorithms implemented in C++ for speed while operating
in numpy arrays.

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


Recent Changes
--------------

For version 0.6.3
~~~~~~~~~~~~~~~~~
- Improve ``mahotas.stretch()`` function
- Fix corner case in surf (when determinant was zero)
- ``threshold`` argument in mahotas.surf
- imreadfromblob() & imsavetoblob() functions
- ``max_points`` argument for mahotas.surf.interest_points()
- Add ``mahotas.labeled.borders`` function

For version 0.6.2
~~~~~~~~~~~~~~~~~

Bugfix release:

- Fix memory leak in _surf
- More robust searching for freeimage
- More functions in mahotas.surf() to retrieve intermediate results
- Improve compilation on Windows (patches by Christoph Gohlke)

0.6.1 (Dec 13 2010)
~~~~~~~~~~~~~~~~~~~

- SURF local features
- Convolution
- mahotas.labeled functions
- just_filter option in edge.sobel()
- Release the GIL in morphological functions


0.6 (Nov 22 2010)
~~~~~~~~~~~~~~~~~

- Improve Local Binary patterns (faster and better interface)
- Much faster erode() (10x faster)
- Faster dilate() (2x faster)
- TAS for 3D images
- Haralick for 3D images
- Fix mahotas.imread for RGBA images

0.5.3 (Oct 29 2010)
~~~~~~~~~~~~~~~~~~~

- Releases GIL when possible
- Added ``imresize()``
- Much improved ``thin()`` function

*Website*: `http://luispedro.org/software/mahotas
<http://luispedro.org/software/mahotas>`_

*API Docs*: `http://packages.python.org/mahotas/
<http://packages.python.org/mahotas/>`_

*Mailing List*: Use the `pythonvision mailing list
<http://groups.google.com/group/pythonvision?pli=1>`_ for questions, bug
submissions, etc.

*Author*: Luis Pedro Coelho (with code by Zachary Pincus [from scikits.image],
Peter J. Verveer [from scipy.ndimage], and Davis King [from dlib]

