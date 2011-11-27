==========================
Frequently Asked Questions
==========================

Why did you not simply contribute to ``scipy.ndimage`` or ``scikits.image``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When I started this project (although it wasn't called mahotas and it was more
of a collection of semi-organised routines than a project), there was no
``scikits.image``.

In the meanwhile, all these projects have very different internal philosophies.
``ndimage`` is old-school scipy, in C, with macros. ``scikits.image`` uses
Cython extensively, while ``mahotas`` uses C++ and templates. I don't want to
use Cython as I find that it is not yet established enough and it cannot (I
believe) be used to write functions that run on multiple types (like with C++
templates). The scipy community does not want to use C++.

I have, on the other hand, taken code from ndimage and ported it to C++ for use
in mahotas. In the process, I feel it is much cleaner code (because you can use
RAII, exceptions, and templates) and I want to keep it that way.

In any case, we all use the same data format: numpy arrays. It is very easy
(trivial, really) to use all the packages together and take whatever functions
you want from each. All the packages use function based interfaces which make
it easy to mix-and-match.

