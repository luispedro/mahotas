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
    if (!PyArg_ParseTuple(args,"OOi", &array, &result, &diagonal)) return NULL;
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
    cooccurence<type>(numpy::aligned_array<long>(result), numpy::aligned_array<type>(array), diagonal);\

        HANDLE_INTEGER_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    Py_RETURN_NONE;
}


PyMethodDef methods[] = {
  {"cooccurence",(PyCFunction)py_cooccurent, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_texture()
  {
    import_array();
    (void)Py_InitModule("_texture", methods);
  }

