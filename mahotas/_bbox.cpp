#include <algorithm>
#include <queue>
#include <vector>
#include <cstdio>
#include <limits>
#include <iostream>

#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] = 
    "Type not understood. "
    "This is caused by either a direct call to _bbox (which is dangerous: types are not checked!) or a bug in bbox.py.\n"
    "If you suspect the latter, please report it to the mahotas developpers.";


template<typename T>
void bbox(numpy::aligned_array<T> array, numpy::index_type* extrema) {
    const unsigned N = array.size();
    typename numpy::aligned_array<T>::iterator pos = array.begin();
    for (unsigned i = 0; i != N; ++i, ++pos) {
        if (*pos) {
            numpy::position where = pos.position();
            for (unsigned j = 0; j != array.ndims(); ++j) {
                extrema[2*j] = std::min<numpy::index_type>(extrema[2*j], where[j]);
                extrema[2*j+1] = std::max<numpy::index_type>(extrema[2*j+1], where[j]+1);
            }
        }
    }
}


template<typename T>
void carray_bbox(const T* array, unsigned N0, unsigned N1, numpy::index_type* extrema) {
    for (unsigned y = 0; y != N0; ++y) {
        for (unsigned x = 0; x != N1; ++x, ++array)
            if (*array) {
                extrema[0] = std::min<numpy::index_type>(extrema[0], y);
                extrema[1] = std::max<numpy::index_type>(extrema[1], y+1);

                extrema[2] = std::min<numpy::index_type>(extrema[2], x);
                extrema[3] = std::max<numpy::index_type>(extrema[3], x+1);
            }
    }
}

PyObject* py_bbox(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args,"O", &array)) return NULL;
    if (!PyArray_Check(array)) { 
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return 0;
    }
    npy_intp dims[1];
    dims[0] = 2 *array->nd;
    PyArrayObject* extrema = (PyArrayObject*)PyArray_SimpleNew(1, dims, numpy::index_type_number);

    if (!extrema) return NULL;
    npy_intp* extrema_v = static_cast<npy_intp*>(PyArray_DATA(extrema));
    for (int j = 0; j != array->nd; ++j) {
        extrema_v[2*j] = PyArray_DIM(array,j);
        extrema_v[2*j+1] = 0;
    }
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
        if (PyArray_ISCARRAY_RO(array) && array->nd == 2) { \
            carray_bbox<type>(static_cast<const type*>(PyArray_DATA(array)), PyArray_DIM(array,0), PyArray_DIM(array, 1), extrema_v); \
        } else { \
            bbox<type>(numpy::aligned_array<type>(array), extrema_v); \
        }

        HANDLE_INTEGER_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    if (extrema_v[1] == 0) {
        for (int j = 0; j != 2 * array->nd; ++j) extrema_v[j] = 0;
    }
    return PyArray_Return(extrema);
}


PyMethodDef methods[] = {
  {"bbox",(PyCFunction)py_bbox, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_bbox()
  {
    import_array();
    (void)Py_InitModule("_bbox", methods);
  }

