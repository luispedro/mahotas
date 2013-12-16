// Part of mahotas. See LICENSE file for License
// Copyright 2008-2013 Luis Pedro Coelho <luis@luispedro.org>
#include "../utils.hpp"
#include "../numpypp/numpy.hpp"

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _lbp (which is dangerous: types are not checked!) or a bug in mahotas.\n";


inline
npy_uint32 roll_right(npy_uint32 v, int points) {
    return (v >> 1) | ((v & 1) << (points-1));
}

inline
npy_uint32 map(npy_uint32 v, int points) {
    npy_uint32 min = v;
    for (int i = 0; i != points; ++i) {
        v = roll_right(v, points);
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
    const int size = PyArray_DIM(array, 0);
    npy_uint32* data = numpy::ndarray_cast<npy_uint32*>(array);
    {
        gil_release nogil;
        // This is not a great implementation, but LBP spends a tiny fraction
        // of its time here.
        for (int i = 0; i != size; ++i) {
            data[i] = map(data[i], npoints);
        }
    }
    Py_INCREF(array);
    return PyArray_Return(array);
}


PyMethodDef methods[] = {
  {"map",(PyCFunction)py_map, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace

DECLARE_MODULE(_lbp)

