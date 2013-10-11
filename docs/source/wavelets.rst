==================
Wavelet Transforms
==================

.. versionadded:: 0.9.1
    Wavelet functions were only added in version 0.9.1

We are going to use wavelets to transform an image so that most of its values
are 0 (and otherwise small), but most of the signal is preserved.

The code for this tutorial is avalailable from the source distribution as
``mahotas/demos/wavelet_compression.py``.

We start by importing and loading our input image

.. plot::
    :context:
    :include-source:

    import numpy as np
    import mahotas
    import mahotas.demos

    from mahotas.thresholding import soft_threshold
    from matplotlib import pyplot as plt
    from os import path
    f = mahotas.demos.load('luispedro', as_grey=True)
    f = f[:256,:256]
    plt.gray()
    # Show the data:
    print("Fraction of zeros in original image: {0}".format(np.mean(f==0)))
    plt.imshow(f)
    plt.show()

There are no zeros in the original image. We now try a baseline compression
method: save every other pixel and only high-order bits.

.. plot::
    :context:
    :include-source:

    direct = f[::2,::2].copy()
    direct /= 8
    direct = direct.astype(np.uint8)
    print("Fraction of zeros in original image (after division by 8): {0}".format(np.mean(direct==0)))
    plt.imshow(direct)
    plt.show()

There are only a few zeros, though. We have, however, thrown away 75% of the
values. Can we get a better image, using the same number of values, though?

We will transform the image using a Daubechies wavelet (D8) and then discard
the high-order bits.

.. plot::
    :context:
    :include-source:

    # Transform using D8 Wavelet to obtain transformed image t:
    t = mahotas.daubechies(f,'D8')

    # Discard low-order bits:
    t /= 8
    t = t.astype(np.int8)
    print("Fraction of zeros in transform (after division by 8): {0}".format(np.mean(t==0)))
    plt.imshow(t)
    plt.show()

This has 60% zeros! What does the reconstructed image look like?

.. plot::
    :context:
    :include-source:


    # Let us look at what this looks like
    r = mahotas.idaubechies(t, 'D8')
    plt.imshow(r)
    plt.show()


This is a pretty good reduction without much quality loss. We can go further and
discard small values in the transformed space. Also, let's make the remaining
values even smaller in magnitude.

Now, this will be 77% of zeros, with the remaining being small values. This
image would compress very well as a lossless image and we could reconstruct the
full image after transmission. The quality is certainly higher than just
keeping every fourth pixel and low-order bits.

.. plot::
    :context:
    :include-source:


    tt = soft_threshold(t, 12)
    print("Fraction of zeros in transform (after division by 8 & soft thresholding): {0}".format(np.mean(tt==0)))
    # Let us look again at what we have:
    rt = mahotas.idaubechies(tt, 'D8')
    plt.imshow(rt)


What About the Borders?
-----------------------

In this example, we can see some artifacts at the border. We can use
``wavelet_center`` and ``wavelet_decenter`` to handle borders to correctly::

    
    fc = mahotas.wavelet_center(f)
    t = mahotas.daubechies(fc, 'D8')
    r = mahotas.idaubechies(fc, 'D8')
    rd = mahotas.wavelet_decenter(r, fc.shape)

Now, ``rd`` is equal (except for rounding) to ``fc`` **without any border effects**.

API Documentation
-----------------

.. automodule:: mahotas
    :members: haar, ihaar, daubechies, idaubechies
    :noindex:

