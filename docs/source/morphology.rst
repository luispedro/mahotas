=======================
Morphological Operators
=======================
.. versionadded:: 0.8
    open() & close() were added in version 0.8

Morphological operators were the first operations in mahotas (back then, it was
even, briefly, just a single C++ module called ``morph``). Since then, mahotas
has grown a lot. This module, too, has grown and acquired more morphological
operators as well as being optimised for speed.

Let us first select an interesting image

.. plot::
    :context:

    import mahotas
    import mahotas.demos
    from pylab import gray, imshow, show
    import numpy as np

    luispedro = mahotas.demos.load('luispedro')
    luispedro = luispedro.max(2)
    T = mahotas.otsu(luispedro)
    lpbin = (luispedro > T)
    gray()
    eye = ~lpbin[112:180,100:190]
    imshow(eye)
    show()


After Oct 2013, you can get this image with mahotas as::

    import mahotas.demos
    luispedro = mahotas.demos.load('luispedro')
    luispedro = luispedro.max(2)

Dilation & Erosion
------------------

`Dilation <http://en.wikipedia.org/wiki/Dilation_(morphology)>`__ and `erosion
<http://en.wikipedia.org/wiki/Erosion_(morphology)>`__ are two very basic
operators (mathematically, you only need one of them as you
can define the erosion as dilation of the negative or vice-versa).

These operations are available in the ``mahotas.morph`` module:

::

    mahotas.morph.dilate(eye)


Dilation is, intuitively, making positive areas "fatter":

.. plot::
    :context:

    imshow(mahotas.morph.dilate(eye))
    show()

::

    mahotas.morph.erode(eye)

Erosion, by contrast, thins them out:

.. plot::
    :context:

    imshow(mahotas.morph.erode(eye))
    show()

Mahotas supports greyscale erosion and dilation (depending on the ``dtype`` of
the arguments) and you can specify any structuring element you wish (including
non-flat ones). By default, a 1-cross is used::

    # if no structure-element is passed, use a cross:
    se = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]], bool)

However, you can use whatever structuring element you want::

    se = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]], bool)
    dilated = mahotas.morph.dilate(eye, se)
    eroded = mahotas.morph.erode(eye, se)

Note that when you pass it a non-boolean array as the first argument, you will
get *grescale erosion*. Mahotas supports full grescale erosion, including
arbitrary, flat or non-flat, structuring elements).

Close & Open
------------

Closing and opening are based on erosion and dilation. Again, they work in
greyscale and can use an arbitrary structure element.

Here is closing:

::

    mahotas.morph.close(eye)

.. plot::
    :context:

    imshow(mahotas.morph.close(eye))
    show()


And here is opening:

::

    mahotas.morph.open(eye)

.. plot::
    :context:

    imshow(mahotas.morph.open(eye))
    show()

Both ``close`` and ``open`` take an optional structuring element as a second
argument::

    mahotas.morph.open(eye, se)


