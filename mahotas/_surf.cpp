#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"

#include <vector>
#include <cmath>

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _surf (which is dangerous: types are not checked!) or a bug in surf.py.\n";

/* SURF: Speeded-Up Robust Features
 *
 * The implementation here borrows from DLIB, which is in turn influenced by
 * the very well documented OpenSURF library and its corresponding description
 * of how the fast Hessian algorithm functions: "Notes on the OpenSURF Library"
 * by Christopher Evans.
 */

template <typename T>
double sum_rect(numpy::aligned_array<T> integral, int y, int x, const int dy, const int dx, int h, int w) {
    int y0 = y + dy - h/2;
    int x0 = x + dx - w/2;
    int y1 = y0 + h;
    int x1 = x0 + w;
    y0 = std::max<T>(y0, 0);
    y1 = std::max<T>(y1, 0);
    x0 = std::min<T>(x0, integral.dim(1));
    x1 = std::min<T>(x1, integral.dim(1));

    const T A = integral.at(y0,x0);
    const T B = integral.at(y0,x1);
    const T C = integral.at(y1,x0);
    const T D = integral.at(y1,x1);

    // This expression, unlike equivalent alternatives,
    // has no overflows. (D > B) and (C > A) and (D-B) > (C-A)
    return (D - B) - (C - A);
}

int get_border_size(const int octave, const int nr_intervals) {
    const double lobe_size = std::pow(2.0, octave+1.0)*(nr_intervals+1) + 1;
    const double filter_size = 3*lobe_size;

    const int bs = static_cast<int>(std::ceil(filter_size/2.0));
    return bs;
}
int get_step_size(const int initial_step_size, const int octave) {
    return initial_step_size*static_cast<int>(std::pow(2.0, double(octave))+0.5);
}

typedef std::vector<numpy::aligned_array<double> > pyramid_type;

template <typename T>
void build_pyramid(numpy::aligned_array<T> integral,
                pyramid_type& pyramid,
                const int nr_octaves,
                const int nr_intervals,
                const int initial_step_size) {
    assert(nr_octaves > 0);
    assert(nr_intervals > 0);
    assert(initial_step_size > 0);

    const int N0 = integral.dim(0);
    const int N1 = integral.dim(1);
    // allocate space for the pyramid
    pyramid.reserve(nr_octaves);
    for (int o = 0; o < nr_octaves; ++o)
    {
        const int step_size = get_step_size(initial_step_size, o);
        pyramid.push_back(numpy::new_array<double>(nr_intervals, N0/step_size, N1/step_size));
        PyArray_FILLWBYTE(pyramid[o].raw_array(), 0);
    }

    // now fill out the pyramid with data
    for (int o = 0; o < nr_octaves; ++o)
    {
        const int step_size = get_step_size(initial_step_size, o);
        const int border_size = get_border_size(o, nr_intervals)*step_size;
        double *pout = pyramid[o].data();

        for (int i = 0; i < nr_intervals; ++i)
        {
            const int lobe_size = static_cast<int>(std::pow(2.0, o+1.0)+0.5)*(i+1) + 1;
            const double area_inv = 1.0/std::pow(3.0*lobe_size, 2.0);
            const int lobe_offset = lobe_size/2+1;

            for (int y = border_size; y < N0 - border_size; y += step_size)
            {
                for (int x = border_size; x < N1 - border_size; x += step_size)
                {
                    double Dxx =     sum_rect(integral, y, x, 0, 0, lobe_size*3, 2*lobe_size-1) -
                                 3.* sum_rect(integral, y, x, 0, 0, lobe_size,   2*lobe_size-1);

                    double Dyy =     sum_rect(integral, y, x, 0, 0, 2*lobe_size-1, lobe_size*3) -
                                 3.* sum_rect(integral, y, x, 0, 0, 2*lobe_size-1, lobe_size);

                    double Dxy = sum_rect(integral, y, x, -lobe_offset, +lobe_offset, lobe_size, lobe_size) +
                                 sum_rect(integral, y, x, +lobe_offset, -lobe_offset, lobe_size, lobe_size) -
                                 sum_rect(integral, y, x, +lobe_offset, +lobe_offset, lobe_size, lobe_size) -
                                 sum_rect(integral, y, x, -lobe_offset, -lobe_offset, lobe_size, lobe_size);

                    // now we normalize the filter responses
                    Dxx *= area_inv;
                    Dyy *= area_inv;
                    Dxy *= area_inv;

                    const double sign_of_laplacian = (Dxx + Dyy < 0) ? -1 : +1;
                    double determinant = Dxx*Dyy - 0.81*Dxy*Dxy;

                    // If the determinant is negative then just blank it out by setting
                    // it to zero.
                    if (determinant < 0) determinant = 0;

                    // Save the determinant of the Hessian into our image pyramid.  Also
                    // pack the laplacian sign into the value so we can get it out later.
                    *pout++ = sign_of_laplacian*determinant;
                }
            }

        }
    }
}

template <typename T>
void integral(numpy::aligned_array<T> array) {
    gil_release nogil;
    const int N0 = array.dim(0);
    const int N1 = array.dim(1);
    if (N0 == 0 || N1 == 0) return;
    for (int j = 1; j != N1; ++j) {
        array.at(0, j) += array.at(0, j - 1);
    }
    for (int i = 1; i != N0; ++i) {
        array.at(i,0) += array.at(i-1,0);
        for (int j = 1; j != N1; ++j) {
            array.at(i,j) += array.at(i-1, j) + array.at(i, j-1) - array.at(i-1, j-1);
        }
    }
}

PyObject* py_pyramid(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    int nr_octaves;
    int nr_intervals;
    int initial_step_size;
    if (!PyArg_ParseTuple(args,"Oiii", &array, &nr_octaves, &nr_intervals, &initial_step_size)) return NULL;
    if (!PyArray_Check(array) || PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    holdref array_ref(array);
    pyramid_type pyramid;
    try {
        switch(PyArray_TYPE(array)) {
        #define HANDLE(type) \
            build_pyramid<type>(numpy::aligned_array<type>(array), pyramid, nr_octaves, nr_intervals, initial_step_size);

            HANDLE_TYPES();
        #undef HANDLE
            default:
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return NULL;
    } catch (const PythonException& exc) {
        PyErr_SetString(exc.type(), exc.message());
        return NULL;
    }
    PyObject* pyramid_list = PyList_New(nr_octaves);
    if (!pyramid_list) return NULL;
    for (int o = 0; o != nr_octaves; ++o) {
        PyObject* arr = reinterpret_cast<PyObject*>(pyramid.at(o).raw_array());
        Py_INCREF(arr);
        PyList_SetItem(pyramid_list, o, arr);
    }
    return pyramid_list;
}


PyObject* py_integral(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args,"O", &array)) return NULL;
    if (!PyArray_Check(array) || PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    Py_INCREF(array);
    switch(PyArray_TYPE(array)) {
    #define HANDLE(type) \
        integral<type>(numpy::aligned_array<type>(array));

        HANDLE_TYPES();
    #undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    return PyArray_Return(array);
}

PyMethodDef methods[] = {
  {"integral",(PyCFunction)py_integral, METH_VARARGS, NULL},
  {"pyramid",(PyCFunction)py_pyramid, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_surf()
  {
    import_array();
    (void)Py_InitModule("_surf", methods);
  }

