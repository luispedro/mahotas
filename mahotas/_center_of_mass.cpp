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
    "This is caused by either a direct call to _center_of_mass (which is dangerous: types are not checked!) or a bug in center_of_mass.py.\n"
    "If you suspect the latter, please report it to the mahotas developpers.";


template<typename T>
double center_of_mass(const numpy::aligned_array<T> array, npy_double* centers) {
    const unsigned N = array.size();
    double total_sum = 0.;
    typename numpy::aligned_array<T>::const_iterator pos = array.begin();
    for (unsigned i = 0; i != N; ++i, ++pos) {
        double val = *pos;
        total_sum += val;
 //       for (int j = 0; j != nd; ++j) {
            centers[0] += val * pos.index_rev(0);
            centers[1] += val * pos.index_rev(1);
   //     }
    }
    return total_sum;
}

PyObject* py_center_of_mass(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args,"O", &array)) return NULL;
    if (!PyArray_Check(array)) { 
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return 0;
    }
    npy_intp dims[1];
    dims[0] = array->nd;
    PyArrayObject* centers = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!centers) return NULL;
    npy_double* centers_v = static_cast<npy_double*>(PyArray_DATA(centers));
    for (int j = 0; j != array->nd; ++j) {
        centers_v[j] = 0;
    }
    double total_sum;
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
        total_sum = center_of_mass<type>(numpy::aligned_array<type>(array), centers_v); \

        HANDLE_INTEGER_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    int nd = array->nd;
    for (int j = 0; j != nd; ++j) {
        centers_v[j] /= total_sum;
    }
    for (int j = 0; j != nd/2; ++j) {
        std::swap(centers_v[j], centers_v[nd - j - 1]);
    }
    return PyArray_Return(centers);
}


PyMethodDef methods[] = {
  {"center_of_mass",(PyCFunction)py_center_of_mass, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_center_of_mass()
  {
    import_array();
    (void)Py_InitModule("_center_of_mass", methods);
  }

