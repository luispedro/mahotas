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
    "This is caused by either a direct call to _surf (which is dangerous: types are not checked!) or a bug in surf.py.\n";

/* SURF: Speeded-Up Robust Features
 *
 * The implementation here borrows from DLIB, which is in turn influenced by
 * the very well documented OpenSURF library and its corresponding description
 * of how the fast Hessian algorithm functions: "Notes on the OpenSURF Library"
 * by Christopher Evans.
 */

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
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_surf()
  {
    import_array();
    (void)Py_InitModule("_surf", methods);
  }

