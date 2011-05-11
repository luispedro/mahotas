=======================
Labeled Image Functions
=======================

Labeled images are integer images where the values correspond to different
regions. I.e., region 1 is all of the pixels which have value *1*, region two
is the pixels with value 2, and so on. By convention, region 0 is the
background and processed differently.

Labeling Images
---------------
.. versionadded:: 0.6.5

The first step is obtaining a labeled function from a binary function:

.. plot::
    :include-source:

    import mahotas
    import numpy as np
    from pylab import imshow, show

    regions = np.zeros((8,8), bool)

    regions[:3,:3] = 1
    regions[6:,6:] = 1
    labeled, nr_objects = mahotas.label(regions)

    imshow(labeled, interpolation='nearest')
    show()

This results in an image with 3 values:

0. background, where the original image was 0
1. for the first region: (0:3, 0:3);
2. for the second region: (6:, 6:).

There is an extra argument to ``label``: the structuring element, which
defaults to a 3x3 cross (or, 4-neighbourhood).

Borders
-------

A border pixel is one where there is more than one region in its neighbourhood
(one of those regions can be the background).

You can retrieve border pixels with either the ``borders()`` function, which
gets all the borders or the ``border()`` (note the singular) which gets only
the border between a single pair of regions. As usual, what neighbour means is
defined by a structuring element, defaulting to a 3x3 cross.

API Documentation
-----------------

The ``mahotas.labeled`` submodule contains the functions mentioned above.
``label()`` is also available as ``mahotas.label``.

.. automodule:: mahotas.labeled
    :members:
    :noindex:

