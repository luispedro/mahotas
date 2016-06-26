==========================
Frequently Asked Questions
==========================

How do I install mahotas with anaconda?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using `conda <http://anaconda.org/>`__, you can
install mahotas from `conda-forge
<https://conda-forge.github.io/>`__ using the following
commands::

    conda config --add channels conda-forge
    conda install mahotas

Who uses mahotas?
~~~~~~~~~~~~~~~~~

In June 2016, there were `34 papers
<https://scholar.google.com/scholar?as_sdt=1,5&hl=en&sciodt=0,5&cites=18199654681754783804&scipsc=>`__
citing the `mahotas paper
<http://dx.doi.org/10.5334/jors.ac>`__

Why did you not simply contribute to ``scipy.ndimage`` or ``scikits.image``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When I started this project (although it wasn't called mahotas and it was more
of a collection of semi-organised routines than a project), there was no
``scikits.image``.

In the meanwhile, all these projects have very different internal philosophies.
``ndimage`` is old-school scipy, in C, with macros. ``scikits.image`` uses
Cython extensively, while ``mahotas`` uses C++ and templates. I don't want to
use Cython as I find that it is not yet established enough and at the time it
could not be used to write functions that run on multiple types (like with C++
templates). The scipy community does not want to use C++.

I have, on the other hand, taken code from ndimage and ported it to C++ for use
in mahotas. In the process, I feel it is much cleaner code (because you can use
RAII, exceptions, and templates) and I want to keep it that way.

In any case, we all use the same data format: numpy arrays. It is very easy
(trivial, really) to use all the packages together and take whatever functions
you want from each. All the packages use function based interfaces which make
it easy to mix-and-match.

I ran out of memory computing Haralick features on 16 bit images. Is it not supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, it is supported, but your machine may not be big enough to do the
computation. In order to compute Haralick features, first a cooccurrence matrix
is computed. This matrix has the size ``(ngrey, ngrey)`` where ``ngrey`` is the
largest grey value in the input. Thus, if your image has a very high dynamic
range (i.e., ``ngrey`` is large), you may not have the resources to compute the
cooccurrence matrix.

It is often a good idea to contrast stretch your images. For example, using the
following code, stretches your images to the 0-255 range::

    im_stretched = mh.stretch(im)
    features = mh.features.haralic(im_stretched)

16 bit images where the dynamic range is not too large (for example, some
imaging equipment can only really produce 12 bits, so ``ngrey < 4096``) are not
a problem.

What are the parameters to Local Binary Patterns?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the documentation on `local binary patterns <lbp.html>`__.

I am using mahotas in a scientific publication, is there a citation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use mahotas in a scientific publication, please cite:

    Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer
    vision. Journal of Open Research Software 1(1), DOI:
    http://dx.doi.org/10.5334/jors.ac

In BibTeX format::

    @article{coelho:mahotas,
        title = {Mahotas: Open source software for scriptable computer vision},
        author = {Luis Pedro Coelho},
        journal = {Journal of Open Research Software},
        year = {2013},
        volume = {1},
        doi = {10.5334/jors.ac},
        url = {http://dx.doi.org/10.5334/jors.ac}
    }

This is accessible in Python using ``mahotas.citation()``.

Imread cannot find FreeImage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mahotas itself does not have the functionality to read in images (see the `I/O
section <io.html>`__.

Functions such as ``imread`` are just a wrapper around one of 2 backends:

1. mahotas-imread (i.e., https://pypi.python.org/pypi/imread)
2. FreeImage

Thus, you need to install one of the packages above. At one point, mahotas
supported wrapping matplotlib, but their image loading methods are unreliable
as it uses other packages itself.  Thus, depending on what you had installed,
the resulting images would be different.

If you are running on Windows, you may wish to try `Christoph Gohlke's packages
<http://www.lfd.uci.edu/~gohlke/pythonlibs/#mahotas>`__.

