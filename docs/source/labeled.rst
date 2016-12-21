=======================
Labeled Image Functions
=======================

Labeled images are integer images where the values correspond to different
regions. I.e., region 1 is all of the pixels which have value *1*, region two
is the pixels with value 2, and so on. By convention, **region 0 is the
background and often handled differently**.

Labeling Images
---------------
.. versionadded:: 0.6.5

The first step is obtaining a labeled function from a binary function:

.. plot::
    :include-source:

    import mahotas as mh
    import numpy as np
    from pylab import imshow, show

    regions = np.zeros((8,8), bool)

    regions[:3,:3] = 1
    regions[6:,6:] = 1
    labeled, nr_objects = mh.label(regions)

    imshow(labeled, interpolation='nearest')
    show()

This results in an image with 3 values:

0. background, where the original image was 0
1. for the first region: (0:3, 0:3);
2. for the second region: (6:, 6:).

There is an extra argument to ``label``: the structuring element, which
defaults to a 3x3 cross (or, 4-neighbourhood). This defines what it means for
two pixels to be in the same region. You can use 8-neighbourhoods by replacing
it with a square::

    labeled,nr_objects = mh.label(regions, np.ones((3,3), bool))

We can now collect a few statistics on the labeled regions. For example, how
big are they?

::

    sizes = mh.labeled.labeled_size(labeled)
    print('Background size', sizes[0])
    print('Size of first region: {}'.format(sizes[1]))

This size is measured simply as the number of pixels in each region. We can
instead measure the total weight in each area::

    array = np.random.random_sample(regions.shape)
    sums = mh.labeled_sum(array, labeled)
    print('Sum of first region: {}'.format(sums[1]))

Filtering Regions
-----------------

.. versionadded:: 0.9.6
    ``remove_regions`` & ``relabel`` were added.

Here is a slightly more complex example. The full code is in the `demos
directory <https://github.com/luispedro/mahotas/tree/master/mahotas/demos>`__
as ``nuclear.py``. We are going to use this image, a fluorescent microscopy
image from a `nuclear segmentation benchmark
<http://luispedro.org/projects/nuclear-segmentation>`__

This image is available as ``mahotas.demos.nuclear_image()``

.. plot::
   :context:

    import mahotas as mh
    import mahotas.demos
    import numpy as np
    from pylab import imshow, show

    f = mh.demos.nuclear_image()
    f = f[:,:,0]
    imshow(f)
    show()

First we perform a bit of Gaussian filtering and thresholding:

::

    f = mh.gaussian_filter(f, 4)
    f = (f> f.mean())

(Without the Gaussian filter, the resulting thresholded image has very noisy
edges. You can get the image in the ``demos/`` directory and try it out.)

.. plot::
   :context:

    f = mh.gaussian_filter(f, 4)
    f = (f> f.mean())
    imshow(f)
    show()

Labeling gets us all of the nuclei::

    labeled, n_nucleus  = mh.label(f)
    print('Found {} nuclei.'.format(n_nucleus))

.. plot::
   :context:

    labeled, n_nucleus  = mh.label(f)
    print('Found {} nuclei.'.format(n_nucleus))
    imshow(labeled)
    show()

``42`` nuclei were found. None were missed, but, unfortunately, we also get
some aggregates. In this case, we are going to assume that we wanted to perform
some measurements on the real nuclei, but are willing to filter out anything
that is not a complete nucleus or that is a lump on nuclei. So we measure sizes
and filter::

    sizes = mh.labeled.labeled_size(labeled)
    too_big = np.where(sizes > 10000)
    labeled = mh.labeled.remove_regions(labeled, too_big)

.. plot::
   :context:

    sizes = mh.labeled.labeled_size(labeled)
    too_big = np.where(sizes > 10000)
    labeled = mh.labeled.remove_regions(labeled, too_big)
    imshow(labeled)
    show()

We can also remove the region touching the border::

    labeled = mh.labeled.remove_bordering(labeled)

.. plot::
   :context:

    labeled = mh.labeled.remove_bordering(labeled)
    imshow(labeled)
    show()

This array, ``labeled`` now has values in the range ``0`` to ``n_nucleus``, but
with some values missing (e.g., if region ``7`` was one of the ones touching
the border, then ``7`` is not used in the labeling). We can ``relabel`` to get
a cleaner version::

    relabeled, n_left = mh.labeled.relabel(labeled)
    print('After filtering and relabeling, there are {} nuclei left.'.format(n_left))

Now, we have ``24`` nuclei and ``relabeled`` goes from ``0`` (background) to ``24``.

.. plot::
   :context:

    relabeled, n_left = mh.labeled.relabel(labeled)
    print('After filtering and relabeling, there are {} nuclei left.'.format(n_left))
    imshow(relabeled)
    show()

In mahotas after version ``1.4``, we can even make many of the same operations
with a single call to ``mh.labeled.filter_labeled``::

    relabeled,n_left = mh.labeled.filter_labeled(labeled, remove_bordering=True, max_size=10000)


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

