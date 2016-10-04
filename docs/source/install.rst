How To Install Mahotas
======================

.. image:: https://anaconda.org/conda-forge/mahotas/badges/installer/conda.svg   :target: https://conda.anaconda.org/conda-forge

The simplest way to install mahotas is to use `conda <http://anaconda.org/>`__.

If you have conda installed, you can install mahotas using the following pair
of commands::

    conda config --add channels conda-forge
    conda install mahotas

This relies on the `conda-forge <https://conda-forge.github.io/>`__ project,
which builds packages for all major environments (Linux, Mac OS X, and
Windows). If you do not want to permanently add the conda-forge channel to your
conda configuration, you can also install just mahotas with::

    conda install -c https://conda.anaconda.org/conda-forge mahotas

From source
-----------

You can get the released version using pip::

    pip install mahotas

If you prefer, you can download the source from `PyPI
<http://pypi.python.org/pypi/mahotas>`__ and run::

    python setup.py install

You will need to have ``numpy`` and a ``C++`` compiler.

Visual Studio
~~~~~~~~~~~~~

For compiling from source in Visual Studio, use::

    python setup.py build_ext -c msvc
    python setup.py install


Bleeding Edge (Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Development happens on `github <https://github.com/luispedro/mahotas>`__. You
can get the development source there. Watch out that *these versions are more
likely to have problems*.

Packaged Versions
-----------------

On Windows
~~~~~~~~~~

On Windows, Christoph Gohlke does an excelent job maintaining `binary packages
of mahotas <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`__ (and several other
packages).


WinPython
~~~~~~~~~

`WinPython <http://winpython.sourceforge.net/>`__ ships with `mahotas as a
standard package <http://sourceforge.net/p/winpython/wiki/PackageIndex_27/>`__

Python(x, y)
~~~~~~~~~~~~

If you use `Python(x, y) <http://python-xy.github.io/>`__, which is often a good
solution, then you probably have it already as `mahotas is a standard plugin
<http://python-xy.github.io>`__.

Enthought Canopy
~~~~~~~~~~~~~~~~

Since May 2015, Enthought's Canopy Package Index `includes mahotas
<https://www.enthought.com/products/canopy/package-index/>`__.

FreeBSD
~~~~~~~

Mahotas is available for FreeBSD as `graphics/mahotas
<http://www.freshports.org/graphics/mahotas>`__.

MacPorts
~~~~~~~~

For Macports, mahotas is available as `py27-mahotas
<https://trac.macports.org/browser/trunk/dports/python/py-mahotas/Portfile>`__.

conda
~~~~~

Mahotas is not a part of standard conda packages, but on 64 bit Linux, you can
get it `from this repository <https://binstar.org/luispedro/mahotas>`__ with::

    conda install -c https://conda.binstar.org/luispedro mahotas


Frugalware Linux
~~~~~~~~~~~~~~~~

Mahotas is available as ``python-mahotas``.

