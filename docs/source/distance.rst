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

.. plot:: ../../mahotas/demos/distance.py
    :include-source:

Distance Transform and Watershed
--------------------------------

The distance transform is often combined with the watershed for segmentation.
Here is an example (which is available with the source in the
``mahotas/demos/`` directory as ``nuclear_distance_watershed.py``).

.. plot:: ../../mahotas/demos/nuclear_distance_watershed.py

The code is not very complex. Start by loading the image and preprocessing it
with a Gaussian blur::

    import mahotas
    import mahotas.demos

    nuclear = mahotas.demos.nuclear_image()
    nuclear = nuclear[:,:,0]
    nuclear = mahotas.gaussian_filter(nuclear, 1.)
    threshed  = (nuclear > nuclear.mean())

Now, we compute the distance transform::

    distances = mahotas.stretch(mahotas.distance(threshed))

We find and label the regional maxima::

    Bc = np.ones((9,9))
    maxima = mahotas.morph.regmax(distances, Bc=Bc)
    spots,n_spots = mahotas.label(maxima, Bc=Bc)

Finally, to obtain the image above, we invert the distance transform (because
of the way that ``cwatershed`` is defined) and compute the watershed::

    surface = (distances.max() - distances)
    areas = mahotas.cwatershed(surface, spots)
    areas *= threshed

We used a random colormap with a black background for the final image. This is
achieved by::

    import random
    from matplotlib import colors as c
    colors = map(cm.jet,range(0, 256, 4))
    random.shuffle(colors)
    colors[0] = (0.,0.,0.,1.)
    rmap = c.ListedColormap(colors)
    imshow(areas, cmap=rmap)
    show()

API Documentation
-----------------

.. automodule:: mahotas
    :members: distance
    :noindex:

