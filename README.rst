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
 - thresholding.
 - convex points calculations.
 - hit & miss. thinning.
 - Zernike & Haralick features.
 - freeimage based numpy image loading (requires freeimage libraries to be
 installed).


Recent Changes
--------------

post 0.6
~~~~~~~~
- Release the GIL in morphological functions
- Convolution


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

