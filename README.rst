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

*Author*: Luis Pedro Coelho (with code by Zachary Pincus [from scikits.image]
and Peter J. Verveer [from scipy.ndimage])

