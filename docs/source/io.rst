=========================
Input/Output with Mahotas
=========================

Mahotas does not have any builtin support for input/output. However, it wraps a
few other libraries that do:

1.  It prefers `imread <https://github.com/luispedro/imread>`__, if it is
    available. Imread is a native C++ library which reads images into Numpy
    arrays. It supports PNG, JPEG, TIFF, and a few microscopy formats (LSM and
    STK).

2.  It also looks for `freeimage <http://freeimage.sourceforge.net/>`__.
    Freeimage can read and write many formats. Unfortunately, it is harder to
    install and it is not as well-maintained as imread.

3.  As a final fallback, it tries to use `matplotlib
    <http://matplotlib.org/>`__, which has builtin PNG support and wraps PIL
    for other formats.

