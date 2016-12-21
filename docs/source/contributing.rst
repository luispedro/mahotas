============
Contributing
============

Development happens on `github <https://github.com/luispedro/mahotas>`__ and
the preferred contribution method is by forking the repository and issuing a
pull request. Alternatively, just sending a patch to luis@luispedro.org will
work just as well.

If you don't know git (or another distributed version control system), which is
a fantastic tool in general, there are several good git and github tutorials.
You can start with their `official documentation <https://help.github.com/>`__.

If you want to work on the C++ code, you can read the chapter in the `internals
<internals.html>`__ before you start. Also, read the `principles
<principles.html>`__ declaration.

Debug Mode
----------

If you compile mahotas in debug mode, then it will run slower but perform a lot
of runtime checks. This is controlled by the ``DEBUG`` environment variable.

There are two levels:

1.  ``DEBUG=1`` This turns on assertions. The code will run slower, but
    probably not noticeably slower, except for very large images.
2.  ``DEBUG=2`` This turns on the assertions and additionally uses the debug
    version of the C++ library (this only works if you are using GCC). Some of
    the internal code also picks up on the ``DEBUG=2`` and adds even more
    sanity checking. The result will be code that runs **much slower** as all
    operations done through iterators into standard containers are now checked
    (including many inner loop operations). However, it catches many errors.

The Makefile that comes with the source helps you::

    make clean
    make debug
    make test

will rebuild in debug mode and run all tests. When you are done testing, use
the ``fast`` Make target to get the non-debug build::

    make clean
    make fast

Using make will not change your environment. The ``DEBUG`` variable is set
internally only.

If you don't know about it, check out `ccache <http://ccache.samba.org/>`__
which is a great tool if you are developing in compiled languages (this is not
specific to mahotas or even Python). It will allow you to quickly perform
``make clean; make debug`` and ``make clean; make fast`` so you never get your
builds mixed up.

