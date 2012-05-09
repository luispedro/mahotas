=======================
Morphological Operators
=======================
.. versionadded:: 0.8
    open() & close() were only added in version 0.8

Morphological operators were the first operations in mahotas (back then, it was
even, briefly, just a single C++ module called ``morph``).


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

Operations
----------

::

    mahotas.morph.dilate(eye)

.. plot::
    :context:

    imshow(mahotas.morph.dilate(eye))
    show()

::

    mahotas.morph.erode(eye)

.. plot::
    :context:

    imshow(mahotas.morph.erode(eye))
    show()

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

