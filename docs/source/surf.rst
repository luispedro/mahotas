==========================
Speeded-Up Robust Features
==========================

.. versionadded:: 0.8
   In version 0.8, some of the inner functions are now in mahotas.features.surf
   instead of mahotas.surf


`Speeded-Up Robust Features (SURF)
<https://en.wikipedia.org/wiki/Speeded_up_robust_features>`__ are a recent
innovation in the *local features* family. There are two steps to this
algorithm:

1. Detection of interest points.
2. Description of interest points.

The function ``mahotas.features.surf.surf`` combines the two steps::

    import numpy as np
    from mahotas.features import surf

    f = ... # input image
    spoints = surf.surf(f)
    print("Nr points: {}".format(len(spoints)))

Given the results, we can perform a simple clustering using, for example, `
`scikit-learn <scikit-learn.org>`__::

    try:
        from sklearn.cluster import KMeans

        # spoints includes both the detection information (such as the position
        # and the scale) as well as the descriptor (i.e., what the area around
        # the point looks like). We only want to use the descriptor for 
        # clustering. The descriptor starts at position 5:
        descrs = spoints[:,5:]
        
        # We use 5 colours just because if it was much larger, then the colours
        # would look too similar in the output.
        k = 5
        values = KMeans(n_clusters=k).fit(descrs).labels_
        colors = np.array([(255-52*i,25+52*i,37**i % 101) for i in xrange(k)])
    except:
        values = np.zeros(100)
        colors = [(255,0,0)]

So we are assigning different colours to each of the possible 

The helper ``surf.show_surf`` draws coloured polygons around the
interest points::

    f2 = surf.show_surf(f, spoints[:100], values, colors)
    imshow(f2)
    show()


Running the above on a photo of luispedro, the author of mahotas yields:

.. plot:: ../../mahotas/demos/surf_luispedro.py
    :include-source:

API Documentation
-----------------

The ``mahotas.features.surf`` module contains separate functions for all the steps in
the SURF pipeline.

.. automodule:: mahotas.features.surf
    :members:
    :noindex:


