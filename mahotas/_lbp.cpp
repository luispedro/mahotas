#include <iostream>
#include "utils.hpp"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _lbp (which is dangerous: types are not checked!) or a bug in mahotas.\n";


inline
npy_uint32 roll_left(npy_uint32 v, int points) {
    return (v >> 1) | ( (1 << (points-1)) * (v & 1) );
}

inline
npy_uint32 map(npy_uint32 v, int points) {
    npy_uint32 min = v;
    for (int i = 0; i != points; ++i) {
        v = roll_left(v, points);
        if (v < min) min = v;
    }
    return min;
}

PyObject* py_map(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    int npoints;
    if (!PyArg_ParseTuple(args,"Oi", &array, &npoints) ||
        !PyArray_Check(array) || PyArray_TYPE(array) != NPY_UINT32  ||
        PyArray_NDIM(array) != 1 || !PyArray_ISCONTIGUOUS(array) ) {
            PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
            return NULL;
    }
    npy_uint32* data = reinterpret_cast<npy_uint32*>(PyArray_DATA(array));
    for (int i = 0; i != PyArray_DIM(array, 0); ++i) {
        data[i] = map(data[i], npoints);
    }
    Py_INCREF(array);
    return PyArray_Return(array);
}


PyMethodDef methods[] = {
  {"map",(PyCFunction)py_map, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_lbp()
  {
    import_array();
    (void)Py_InitModule("_lbp", methods);
  }

