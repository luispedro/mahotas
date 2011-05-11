==========================
Speeded-Up Robust Features
==========================

.. versionadded:: 0.6.1
   SURF is only available starting in version 0.6.1, with an important bugfix
   in version 0.6.2.


Speeded-Up Robust Features (SURF) are a recent innnovation in the *local
features* family. There are two steps to this algorithm:

1. Detection of interest points.
2. Description of interest points.

The function ``mahotas.surf.surf`` combines the two steps::

    import numpy as np
    import mahotas.surf

    f = ... # input image
    spoints = mahotas.surf.surf(f)
    print "Nr points:", len(spoints)

Given the results, we can perform a simple clustering using, for example, `milk
<http://luispedro.org/software/milk>`__ (we could have used any other system,
of course; having written milk, I am most familiar with it)::

    try:
        import milk

        # spoints includes both the detection information (such as the position
        # and the scale) as well as the descriptor (i.e., what the area around
        # the point looks like). We only want to use the descriptor for 
        # clustering. The descriptor starts at position 5:
        descrs = spoints[:,5:]
        
        # We use 5 colours just because if it was much larger, then the colours
        # would look too similar in the output.
        k = 5
        values, _  = milk.kmeans(descrs, k)
        colors = np.array([(255-52*i,25+52*i,37**i % 101) for i in xrange(k)])
    except:
        values = np.zeros(100)
        colors = [(255,0,0)]

So we are assigning different colours to each of the possible 

The helper ``mahotas.surf.show_surf`` draws coloured polygons around the
interest points::

    f2 = mahotas.surf.show_surf(f, spoints[:100], values, colors)
    imshow(f2)
    show()


Running the above on a photo of luispedro, the author of mahotas yields:

.. plot:: mahotas/demos/surf_luispedro.py
    :include-source:

API Documentation
-----------------

The ``mahotas.surf`` module contains separate functions for all the steps in
the SURF pipeline.

.. automodule:: mahotas.surf
    :members:
    :noindex:
