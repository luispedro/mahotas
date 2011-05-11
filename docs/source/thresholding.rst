============
Thresholding
============

The example in this section is present in the source under
``mahotas/demos/thresholding.py``.

We start with an image, a grey-scale image::


    luispedro_image = 'mahotas/demos/data/luispedro.jpg'
    photo = mahotas.imread(luispedro_image, as_grey=True)
    photo = photo.astype(np.uint8)

The reason we convert to ``np.uint8`` is because ``as_grey`` returns floating
point images (there are good reasons for this and good reasons against it).

.. plot::

    import mahotas
    import numpy as np
    from pylab import imshow, gray, show
    from os import path

    luispedro_image = 'mahotas/demos/data/luispedro.jpg'
    photo = mahotas.imread(luispedro_image, as_grey=True)
    photo = photo.astype(np.uint8)

    gray()
    imshow(photo)
    show()

Thresholding functions have a trivial interface: they take an image and return
a value. One of the most well-known thresholding methods is Otsu's method::

    T_otsu = mahotas.otsu(photo)
    print T_otsu
    imshow(photo > T_otsu)
    show()

prints ``115``.

.. plot::

    import mahotas
    import numpy as np
    from pylab import imshow, gray, show
    from os import path

    luispedro_image = 'mahotas/demos/data/luispedro.jpg'
    photo = mahotas.imread(luispedro_image, as_grey=True)
    photo = photo.astype(np.uint8)


    T_otsu = mahotas.otsu(photo)
    print T_otsu
    gray()
    imshow(photo > T_otsu)
    show()

An alternative is the Riddler-Calvard method::

    T_rc = mahotas.rc(photo)
    print T_rc
    imshow(photo > T_rc)
    show()

In this image, it prints almost the same as Otsu: ``115.68``. The thresholded
image is exactly the same:

.. plot::

    import mahotas
    import numpy as np
    from pylab import imshow, gray, show
    from os import path

    luispedro_image = 'mahotas/demos/data/luispedro.jpg'
    photo = mahotas.imread(luispedro_image, as_grey=True)
    photo = photo.astype(np.uint8)


    T_rc = mahotas.rc(photo)
    print T_rc
    gray()
    imshow(photo > T_rc)
    show()

API Documentation
-----------------

The ``mahotas.thresholding`` module contains the thresholding functions, but
they are also available in the main ``mahotas`` namespace.

.. automodule:: mahotas.thresholding
    :members:
    :noindex:
