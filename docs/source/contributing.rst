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
of runtime checks.

There are two levels:

1.  A release build with ``b_ndebug=false`` turns on assertions. The code will run slower, but
    probably not noticeably slower, except for very large images.
2.  ``make debug`` uses an optimized release build with assertions enabled and
    additionally defines ``_GLIBCXX_DEBUG``. This only has an effect with
    libstdc++, but when available it enables checked iterators in the C++
    standard library. The result can still be **much slower** as many iterator
    operations are now checked. However, it catches many errors.

The Makefile that comes with the source helps you::

    make clean
    make debug
    make tests

will rebuild in debug mode and run all tests. When you are done testing, use
the ``fast`` Make target to get the non-debug build::

    make clean
    make fast

Using make will rebuild the editable install in your current Python
environment with the requested configuration.

If you don't know about it, check out `ccache <https://ccache.samba.org/>`__
which is a great tool if you are developing in compiled languages (this is not
specific to mahotas or even Python). It will allow you to quickly perform
``make clean; make debug`` and ``make clean; make fast`` so you never get your
builds mixed up.
