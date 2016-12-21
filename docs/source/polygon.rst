=================
Polygon Utilities
=================

Drawing
-------

Mahotas is not a package to generate images, but there are a few simple
functions to draw lines and polygons on an image (the target image is known as
the *canvas* in this documentation).

The simplest function is ``line``: Give it two points and it draws a line
between them. The implementation is simple, and in Python, so it will be slow
for many complex usage.

The main purpose of these utilities is to aid debugging and visualisation. If
you need to generate fancy graphs, look for packages such as `matplotlib
<http://matplotlib.org>`__.

Convex Hull
-----------

Convex hull functions are a more typical image processing feature. Mahotas has
a simple one, called ``convexhull``. Given a boolean image (or anything that
will get interpreted as a boolean image), it finds the convex hull of all its
on points.

The implementation is in C++, so it is fast.

A companion function ``fill_convexhull`` returns the convex hull as a binary
image.

API Documentation
-----------------

.. automodule:: mahotas.polygon
    :members:
    :noindex:

