=========================
Input/Output with Mahotas
=========================

Mahotas does not have any builtin support for input/output. However, it wraps a
few other libraries that do. The result is that you can do::

    import mahotas as mh
    image = mh.imread('file.png')
    mh.imsave('copy.png', image)

It can use the following backends (it tries them in the following order):

1.  It prefers `mahotas-imread <https://github.com/luispedro/imread>`__, if it is
    available. Imread is a native C++ library which reads images into Numpy
    arrays. It supports PNG, JPEG, TIFF, WEBP, BMP, and a few TIFF-based
    microscopy formats (LSM and STK).

2.  It also looks for `freeimage <http://freeimage.sourceforge.net/>`__.
    Freeimage can read and write many formats. Unfortunately, it is harder to
    install and it is not as well-maintained as imread.

3.  Finally, it tries to load `pillow <https://pillow.readthedocs.io/>`__.

Thus, to use the ``imread`` or ``imsave`` functions, you need to install one of
the packages above. At one point, mahotas supported wrapping matplotlib, but
their image loading methods are unreliable as it uses other packages itself.
Thus, depending on what you had installed, the resulting images would be
different.

If you are running on Windows, you may wish to try `Christoph Gohlke's packages
<http://www.lfd.uci.edu/~gohlke/pythonlibs/#mahotas>`__.
