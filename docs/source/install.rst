======================
How To Install Mahotas
======================

From source
-----------

You can get the released version using your favorite Python package manager::

    easy_install mahotas

or::

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

If you use `Python(x, y) <http://pythonxy.com/>`__, which is often a good
solution, then you probably have it already as `mahotas is a standard plugin
<https://code.google.com/p/pythonxy/wiki/StandardPlugins>`__.

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

