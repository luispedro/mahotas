==============
Possible Tasks
==============


Here are a few ideas for improving mahotas.

New Features
------------

- `HOG <http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients>`__
- `BRISK <http://savvash.blogspot.pt/2011/08/brisk-binary-robust-invariant-scalable.html>`__
- `Canny edge detection <http://en.wikipedia.org/wiki/Canny_edge_detector>`__
- `Hough Transform <http://en.wikipedia.org/wiki/Hough_transform>`__
- `bilateral filtering <http://en.wikipedia.org/wiki/Bilateral_filter>`__
- `Non Local Filtering <http://en.wikipedia.org/wiki/Non-local_means>`__
- `Wiener filtering <http://en.wikipedia.org/wiki/Wiener_filter>`__

Small Improvements
------------------

- something like the ``overlay`` function from `pymorph <http://luispedro.org/software/pymorph>`__ (or even just copy it over and adapt it to mahotas style).
- H-maxima transform (again, pymorph can provide a basis)
- `entropy thresholding <http://en.wikipedia.org/wiki/Thresholding_(image_processing)>`__

Internals
---------

These can be very complex as they require an understanding of the inner
workings of mahotas, but that does appeal to a certain personality.

- special case 1-D convolution on C-Arrays in C++. The idea is that you can
  write a tight inner loop in one dimension::

    void multiply(floating* r, const floating* f, const floating a, const int n, const int r_step, const int f_step) {
        for (int i = 0; i != n; ++i) {
            *r += a * *f;
            r += r_step;
            f += f_step;
        }
    }

to implement::

    r[row] += a* f[row+offset]

and you can call this with all the different values of ``a`` and ``offset``
that make up your filter. This would be useful for Guassian filtering.


Tutorials
---------

Mahotas has very good API documentation, but not so many *start to finish*
tutorials which touch several parts of it (and even other packages, the ability
to seamlessly use other packages in Python is, of course, a good reason to use
it).

