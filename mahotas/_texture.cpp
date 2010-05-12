#include <algorithm>
#include <queue>
#include <vector>
#include <cstdio>
#include <limits>
#include <iostream>
#include <cstring>

#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _texture (which is dangerous: types are not checked!) or a bug in texture.py.\n";


template<typename T>
void cooccurence(numpy::aligned_array<long> res, numpy::aligned_array<T> array, int diagonal) {
    using numpy::position;
    const T* ptr = array.data();
    int delta = array.stride(1);
    int nr_rows = array.size(0);
    int nr_cols = array.size(1)-1;
    int stride = array.stride(1);
    int row_step = array.stride(0) - nr_cols*stride;

    //std::cout << "nr_rows: " << nr_rows << '\n';
    //std::cout << "nr_cols: " << nr_cols << '\n';
    //std::cout << "stride: " << stride << '\n';
    //std::cout << "row_step: " << row_step << '\n';
    if (diagonal == 1) {
        --nr_rows;
        delta += array.stride(0);
    }

    for (int row = 0; row != nr_rows; ++row) {
        for (int col = 0; col != nr_cols; ++col) {
            T val = *ptr;
            T val2 = *(ptr + delta);
            ++res.at(npy_intp(val), npy_intp(val2));
            ptr += stride;
        }
        ptr += row_step;
    }
}


PyObject* py_cooccurent(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* result;
    int diagonal;
    int symmetric;
    if (!PyArg_ParseTuple(args,"OOii", &array, &result, &diagonal, &symmetric)) return NULL;
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
    cooccurence<type>(numpy::aligned_array<long>(result), numpy::aligned_array<type>(array), diagonal);\

        HANDLE_INTEGER_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    if (symmetric) {
        numpy::aligned_array<long> cmatrix(result);
        const int s0 = cmatrix.size(0);
        const int s1 = cmatrix.size(0);

        if (s0 != s1) {
            PyErr_SetString(PyExc_RuntimeError, "mahotas._texture.cooccurence: Results matrix not square.");
            return NULL;
        }
        for (int y = 0; y != s0; ++y) {
            for (int x = y; x < s1; ++x) {
                long total = cmatrix.at(y,x) + cmatrix.at(x,y);
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
    if (p.size(1) != unsigned(N)) {
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

