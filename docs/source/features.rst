========
Features
========

By features we mean, basically, numerical functions of the image. That is, any
method that gives me a number from the image, I can call it a *feature*.
Ideally, these should be meaningful.

We can classify features into two types:

global
    These are a function of the whole image.

local
    These **have a position** and are a function of a local image region.

Mahotas supports both types.

Global features
---------------

Haralick features
~~~~~~~~~~~~~~~~~

These are texture features, based on the adjancency matrix (the adjacency
matrix stores in position *(i,j)* the number of times that a pixel takes the
value *i* **next to** a pixel with the value *j*. Given different ways to
define **next to**, you obtain slightly different variations of the features.
Standard practice is to average them out across the directions to get some
rotational invariance.

They can be computed for 2-D or 3-D images and are available in the
``mahotas.features.haralick`` module.

Local Binary Patterns
~~~~~~~~~~~~~~~~~~~~~

Local binary patterns (LBP) are a more recent set of features. Each pixel is
looked at individually. Its neighbourhood is analysed and summarised by a
single numeric code. The normalised histogram across all the pixels in the
image is the final set of features.

Again, this is an attempt at capturing texture. LBPs are insensitive to
orientation and to illumination (scaling).

Threshold Adjancency Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Threshold adjancency statistics (TAS) are a recent innovation too. In the
original version, they have fixed parameters, but we have adapted them to
*parameter-free* versions (see `Structured Literature Image Finder: Extracting
Information from Text and Images in Biomedical Literature
<http://dx.doi.org/10.1007/978-3-642-13131-8_4>`__ by Coelho et al. for a
reference). Mahotas supports both.

Zernike Moments
~~~~~~~~~~~~~~~

Zernike moments are **not** a texture feature, but rather a global measure of
how the mass is distributed.

Local features
--------------

SURF: Speeded-Up Robust Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Speeded-Up Robust Features (SURF) have both a *location* (pixel coordinates)
and a scale (natural size) as well as a descriptor (the local features).

