=======================
Morphological Operators
=======================
.. versionadded:: 0.8
    open() & close() were only added in version 0.8

Morphological operators were the first operations in mahotas (back then, it was
even, briefly, just a single C++ module called ``morph``). Since then, mahotas
hsa grown a lot, including having more morphological operators.


Let's first select an interesting image

.. plot::
    :context:

    import mahotas
    from pylab import gray, imshow, show
    import numpy as np

    luispedro = mahotas.imread('../../mahotas/demos/data/luispedro.jpg')
    luispedro = luispedro.max(2)
    T = mahotas.otsu(luispedro)
    lpbin = (luispedro > T)
    gray()
    eye = ~lpbin[112:180,100:190]
    imshow(eye)
    show()


Dilation & Erosion
------------------

`Dilation <http://en.wikipedia.org/wiki/Dilation_(morphology)>`__ and `erosion
<http://en.wikipedia.org/wiki/Erosion_(morphology)>`__ are two very basic
operators (mathematically, you only need one of them as you
can define the erosion as dilation of the negative or vice-versa).

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
            [0, 1, 0]])


Close & Open
------------

Closing and opening are based on erosion and dilation. Again, they work in
greyscale and can use an arbitrary structure element.

::

    mahotas.morph.close(eye)

.. plot::
    :context:

    imshow(mahotas.morph.close(eye))
    show()

::

    mahotas.morph.open(eye)

.. plot::
    :context:

    imshow(mahotas.morph.open(eye))
    show()

