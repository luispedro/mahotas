============
Thresholding
============

The example in this section is present in the source under
``mahotas/demos/thresholding.py``.

We start with an image, a grey-scale image::

    import mahotas.demos
    photo = mahotas.demos.load('luispedro')
    photo = photo.astype(np.uint8)

Before Oct 2013, the ``mahotas.demos.load`` function did not exist and you
needed to specify the path explicitly::

    luispedro_image = '../../mahotas/demos/data/luispedro.jpg'
    photo = mahotas.imread(luispedro_image, as_grey=True)

The reason we convert to ``np.uint8`` is because ``as_grey`` returns floating
point images (there are good reasons for this and good reasons against it,
since it's easier to truncate than to go back, it returns ``np.uint8``).

.. plot::

    import mahotas
    import mahotas.demos
    import numpy as np
    from pylab import imshow, gray, show
    from os import path

    photo = mahotas.demos.load('luispedro', as_grey=True)
    photo = photo.astype(np.uint8)

    gray()
    imshow(photo)
    show()

Thresholding functions have a trivial interface: they take an image and return
a value. One of the most well-known thresholding methods is Otsu's method::

    T_otsu = mahotas.otsu(photo)
    print(T_otsu)
    imshow(photo > T_otsu)
    show()

prints ``115``.

.. plot::

    import mahotas
    import mahotas.demos
    import numpy as np
    from pylab import imshow, gray, show
    from os import path

    photo = mahotas.demos.load('luispedro', as_grey=True)
    photo = photo.astype(np.uint8)


    T_otsu = mahotas.otsu(photo)
    print(T_otsu)
    gray()
    imshow(photo > T_otsu)
    show()

An alternative is the Riddler-Calvard method::

    T_rc = mahotas.rc(photo)
    print(T_rc)
    imshow(photo > T_rc)
    show()

In this image, it prints almost the same as Otsu: ``115.68``. The thresholded
image is exactly the same:

.. plot::

    import mahotas
    import mahotas.demos
    import numpy as np
    from pylab import imshow, gray, show
    from os import path

    photo = mahotas.demos.load('luispedro', as_grey=True)
    photo = photo.astype(np.uint8)


    T_rc = mahotas.rc(photo)
    print(T_rc)
    gray()
    imshow(photo > T_rc)
    show()

See also the `labeled documentation <labeled.html>`__ which can be very helpful
in combination with thresholding.

API Documentation
-----------------

The ``mahotas.thresholding`` module contains the thresholding functions, but
they are also available in the main ``mahotas`` namespace.

.. automodule:: mahotas.thresholding
    :members:
    :noindex:
