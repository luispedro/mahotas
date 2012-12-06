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

- `entropy thresholding <http://en.wikipedia.org/wiki/Thresholding_(image_processing)>`__


- special case 1-D convolution on C-Arrays in C++. The idea is that you can
write a tight inner loop::

    void (floating* r, const floating* f, const floating a, const int n, const int r_step, const int f_step) {
        for (int i = 0; i != n; ++i) {
            *r = a * *f;
            r += r_step;
            f += f_step;
        }
    }

This would be useful for Guassian filtering.

(This is not really a junior job as it can be complicated to understand all of the inner workings).


