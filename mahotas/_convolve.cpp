#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"
#include "_filters.h"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _convolve (which is dangerous: types are not checked!) or a bug in convolve.py.\n";
const char OutputErrorMsg[] =
    "Output type is not valid. "
    "This is caused by either a direct call to _convolve (which is dangerous: types are not checked!) or a bug in convolve.py.\n";


template<typename T>
void convolve(numpy::aligned_array<T> array, numpy::aligned_array<T> filter, numpy::aligned_array<T> result, int mode) {
    gil_release nogil;
    const int N = array.size();
    typename numpy::aligned_array<T>::iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), filter.raw_array(), ExtendMode(mode), true);
    const int N2 = fiter.size();
    T* out = result.data();

    for (int i = 0; i != N; ++i, fiter.iterate_with(iter), ++iter, ++out) {
        // The reasons for using double instead of T:
        //   (1) it is slightly faster (10%)
        //   (2) it handles over/underflow better
        //   (3) scipy.ndimage.convolve does it
        // 
        // Alternatively, we could have written:
        // T cur = T();
        // 
        // and removed the double cast in double(val)*fiter[j] below.
        double cur = 0.;
        for (int j = 0; j != N2; ++j) {
            T val;
            if (fiter.retrieve(iter, j, val)) {
                cur += double(val)*fiter[j];
            }
        }
        *out = T(cur);
    }
}


PyObject* py_convolve(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    PyArrayObject* output;
    int mode;
    if (!PyArg_ParseTuple(args,"OOOi", &array, &filter, &output, &mode)) return NULL;
    if (!PyArray_Check(array) || !PyArray_Check(filter) || PyArray_TYPE(array) != PyArray_TYPE(filter)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }

    if (reinterpret_cast<PyObject*>(output) == Py_None) {
        output = reinterpret_cast<PyArrayObject*>(
                PyArray_EMPTY(PyArray_NDIM(array), PyArray_DIMS(array), PyArray_TYPE(array), 0));
        if (!output) return NULL;
    } else {
        if (!PyArray_Check(output) ||
            PyArray_NDIM(output) != PyArray_NDIM(array) ||
            PyArray_TYPE(output) != PyArray_TYPE(array) ||
            !PyArray_ISCARRAY(output)) {
            PyErr_SetString(PyExc_RuntimeError, OutputErrorMsg);
            return NULL;
        }
        for (int d = 0; d != PyArray_NDIM(array); ++d) {
            if (PyArray_DIM(array, d) != PyArray_DIM(output, d)) {
                PyErr_SetString(PyExc_RuntimeError, OutputErrorMsg);
                return NULL;
            }
        }
        Py_INCREF(output);
    }

    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
        convolve<type>(numpy::aligned_array<type>(array), numpy::aligned_array<type>(filter), numpy::aligned_array<type>(output), mode);
        HANDLE_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    return PyArray_Return(output);
}

PyMethodDef methods[] = {
  {"convolve",(PyCFunction)py_convolve, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_convolve()
  {
    import_array();
    (void)Py_InitModule("_convolve", methods);
  }

