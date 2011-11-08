#include <algorithm>
#include <queue>
#include <vector>
#include <cstdio>
#include <limits>
#include <iostream>
#include <cstring>
#include <signal.h>

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
    "This is caused by either a direct call to _texture (which is dangerous: types are not checked!) or a bug in texture.py.\n";


template<typename T>
void cooccurence(numpy::aligned_array<npy_int32> res, numpy::aligned_array<T> array, numpy::aligned_array<T> Bc) {
    gil_release nogil;
    const int N = array.size();
    typename numpy::aligned_array<T>::iterator iter = array.begin();
    filter_iterator<T> filter(array.raw_array(), Bc.raw_array(), EXTEND_CONSTANT, true);

    for (int i = 0; i != N; ++i, filter.iterate_with(iter), ++iter) {
        T val = *iter;
        T val2 = 0;
        if(filter.retrieve(iter, 0, val2)) {
            ++res.at(npy_intp(val), npy_intp(val2));
        }
    }
}


PyObject* py_cooccurent(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* result;
    PyArrayObject* Bc;
    int symmetric;
    if (!PyArg_ParseTuple(args,"OOOi", &array, &result, &Bc, &symmetric)) return NULL;
    if (!PyArray_Check(array) || !PyArray_Check(result) || !PyArray_Check(Bc)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    if (PyArray_TYPE(result) != NPY_INT32) {
        PyErr_SetString(PyExc_RuntimeError, "mahotas._texture: expected NPY_INT32 for result array. Do not call _texture.cooccurence directly. It is dangerous!");
        return NULL;
    }

#define HANDLE(type) \
    cooccurence<type>(numpy::aligned_array<npy_int32>(result), numpy::aligned_array<type>(array), numpy::aligned_array<type>(Bc));
    SAFE_SWITCH_ON_INTEGER_TYPES_OF(array, true)
#undef HANDLE
    if (symmetric) {
        numpy::aligned_array<npy_int32> cmatrix(result);
        const int s0 = cmatrix.size(0);
        const int s1 = cmatrix.size(0);

        if (s0 != s1) {
            PyErr_SetString(PyExc_RuntimeError, "mahotas._texture.cooccurence: Results matrix not square.");
            return NULL;
        }
        for (int y = 0; y != s0; ++y) {
            for (int x = y; x < s1; ++x) {
                npy_int32 total = cmatrix.at(y,x) + cmatrix.at(x,y);
                cmatrix.at(y,x) = total;
                cmatrix.at(x,y) = total;
            }
        }
    }
    Py_RETURN_NONE;
}
PyObject* py_compute_plus_minus(PyObject* self, PyObject* args) {
    PyArrayObject* p_;
    PyArrayObject* px_plus_y_;
    PyArrayObject* px_minus_y_;
    if (!PyArg_ParseTuple(args,"OOO", &p_, &px_plus_y_, &px_minus_y_)) return NULL;
    numpy::aligned_array<double> p(p_);
    numpy::aligned_array<double> px_plus_y(px_plus_y_);
    numpy::aligned_array<double> px_minus_y(px_minus_y_);
    const int N = p.size(0);
    if (p.size(1) != N) {
        PyErr_SetString(PyExc_RuntimeError, "compute_plus_minus: p is not square.");
        return NULL;
    }
    for (int i = 0; i != N; ++i) {
        for (int j = 0; j != N; ++j) {
            px_plus_y.at(i+j) += p.at(i,j);
            px_minus_y.at(std::abs(i-j)) += p.at(i,j);
        }
    }

    Py_RETURN_NONE;
}


PyMethodDef methods[] = {
  {"cooccurence",(PyCFunction)py_cooccurent, METH_VARARGS, NULL},
  {"compute_plus_minus",(PyCFunction)py_compute_plus_minus, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_texture()
  {
    import_array();
    (void)Py_InitModule("_texture", methods);
  }

