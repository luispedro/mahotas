==================
Distance Transform
==================

The example in this section is present in the source under
``mahotas/demos/distance.py``.

We start with an image, a black&white image that is mostly black except for two
white spots::


    import numpy as np
    import mahotas

    f = np.ones((256,256), bool)
    f[200:,240:] = False
    f[128:144,32:48] = False

.. plot::

    from pylab import imshow, gray, show
    import numpy as np

    f = np.ones((256,256), bool)
    f[200:,240:] = False
    f[128:144,32:48] = False

    gray()
    imshow(f)
    show()

There is a simple ``distance()`` function which computes the distance map::

    import mahotas
    dmap = mahotas.distance(f)

Now ``dmap[y,x]`` contains the squared euclidean distance of the pixel *(y,x)*
to the nearest black pixel in ``f``. If ``f[y,x] == True``, then ``dmap[y,x] ==
0``.

.. plot:: mahotas/demos/distance.py
    :include-source:

API Documentation
-----------------

.. automodule:: mahotas
    :members: distance
    :noindex:

