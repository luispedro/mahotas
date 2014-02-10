=================
Mahotas Internals
=================

This section is of interest if you are trying to understand how mahotas works
in order to fix something, extend it (patches are always welcome), or use some
of its technology in your projects.

Philosophy
----------

Mahotas should not suck.

This is my main development goal and, if I achieve it, this alone should put
mahotas in the top ten to one percent of software packages.

Mahotas should have no bugs. None. Ever.

Of course, some creep in. So, we settle for the next best thing: *Mahotas
should have no **known bugs***.  Whenever a bug is discovered, the top priority
is to squash it.

Read the `principles of mahotas <principles.html>`__


C++/Python Division
-------------------

Mahotas is, for the most part, written in C++, but almost always, you call a
Python function which checks types and then calls the internal function. This
is slightly slower, but it is easier to develop this way (and, for all but the
smallest image, it will not matter).

So each ``module.py`` will have its associated ``_module.cpp``.

C++ Templates
-------------

The main reason that mahotas is in C++ (and not in pure C) is to use templates.
Almost all C++ functions are actually 2 functions:

1. A ``py_function`` which uses the Python C/API to get arguments, &c. This is
   almost always pure C.
2. A template ``function<dtype>`` which works for the ``dtype`` performing the
   actual operation.

So, for example, this is how *erode* is implemented. ``py_erode`` is generic::

    PyObject* py_erode(PyObject* self, PyObject* args) {
        PyArrayObject* array;
        PyArrayObject* Bc;
        if (!PyArg_ParseTuple(args,"OO", &array, &Bc)) return NULL;
        PyArrayObject* res_a = (PyArrayObject*)PyArray_SimpleNew(array->nd,array->dimensions,PyArray_TYPE(array));
        if (!res_a) return NULL;
        PyArray_FILLWBYTE(res_a, 0);
    #define HANDLE(type) \
        erode<type>(numpy::aligned_array<type>(res_a), numpy::aligned_array<type>(array), numpy::aligned_array<type>(Bc));\

            SAFE_SWITCH_ON_INTEGER_TYPES_OF(array)
    #undef HANDLE
        ...


These functions normally contain a lot of boiler-plate code: read the
arguments, perform some sanity checks, perhaps a bit of initialisation, and
then, the switch on the input type with the help of the
``SAFE_SWITCH_ON_INTEGER_TYPES_OF()`` and friends, which call the right
specialisation of the template that does the actual work. In this example
``erode`` implements (binary) erosion::

    template<typename T>
    void erode(numpy::aligned_array<T> res, numpy::aligned_array<T> array, numpy::aligned_array<T> Bc) {
        gil_release nogil;
        const unsigned N = res.size();
        typename numpy::aligned_array<T>::iterator iter = array.begin();
        filter_iterator<T> filter(res.raw_array(), Bc.raw_array());
        const unsigned N2 = filter.size();
        T* rpos = res.data();

        for (int i = 0; i != N; ++i, ++rpos, filter.iterate_with(iter), ++iter) {
            for (int j = 0; j != N2; ++j) {
                T arr_val = false;
                filter.retrieve(iter, j, arr_val);
                if (filter[j] && !arr_val) goto skip_this_one;
            }
            *rpos = true;
            skip_this_one: continue;
        }
    }

The template machinery is not that complicated and the functions using it are
very simple and easy to read. The only downside is that there is some expansion
of code size. Given the small size of these functions however, this is not a
big issue.

In the snippet above, you can see some other C++ machinery:

``gil_release``
    This is a RAII object that release the GIL in its constructor and gets it
    back in its destructor. Normally, the template function will release the
    GIL after the Python-specific code is done.
``array``
    This is a thin wrapper around ``PyArrayObject`` that knows its type and has
    iterators. Relying on these objects has the further advantage that in debug
    mode, it checks bounds for many memory accesses. While this is very costly
    for everyday usage, it can catch bugs faster than the alternatives.
``filter_iterator``
    This is taken from ``scipy.ndimage`` and it is useful to iterate over an
    image and use a centered filter around each pixel (it keeps track of all of
    the boundary conditions).

The inner loop is as direct an implementation of erosion as one would wish for:
for each pixel in the image, look at its neighbours. If all are true, then set
the corresponding output pixel to ``true`` (else, skip it as it has been
initialised to zero).

Most of the functions follow this architecture.

