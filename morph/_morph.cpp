#include "array.hpp"
#include "dispatch.hpp"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}
#include <cstdio>

using namespace numpy_utils;

template <typename T>
numpy_position central_position(const numpy_array_base<T>& array) {
    numpy_position res(array.raw_dims(), array.ndims());
    for (int i = 0, nd = array.ndims(); i != nd; ++i) res.position[i] /= 2;
    return res;
}

template<typename T>
void erode(numpy_aligned_array<T> res, numpy_array<T> array, numpy_aligned_array<T> Bc) {
    const unsigned N = res.size();
    const unsigned N2 = Bc.size();

    typename numpy_array<T>::iterator pos = array.begin();
    for (int i = 0; i != N; ++i, ++pos) {
        typename numpy_aligned_array<T>::iterator startc = Bc.begin();
        for (int j = 0; j != N2; ++j, ++startc) {
            numpy_position npos = pos.position() + startc.position();
            if (res.validposition(npos)) {
                res.at(npos) = *pos+*startc;
            }
        }
    }
}


PyObject* py_erode(PyObject* self, PyObject* args, PyObject* kwds) {
    PyArrayObject* array;
    PyArrayObject* Bc;
    static char * kwlist[] = { "array", "Bc", NULL };
    if (!PyArg_ParseTupleAndKeywords(args,kwds,"OO",kwlist,
                    &array,
                    &Bc)) {
        return NULL;
    }
    PyArrayObject* res_a = (PyArrayObject*)PyArray_FromDims(array->nd,array->dimensions,PyArray_TYPE(array));
    if (!res_a) { 
        return NULL;
    }
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
    erode<type>(numpy_aligned_array<type>(res_a),numpy_array<type>(array),numpy_aligned_array<type>(Bc));\

        HANDLE_INTEGER_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError,"Type not understood. This is caused by either a direct call to _morph (which is dangerous: types are not checked!) or a bug in morph.py.\n");
        return NULL;
    }
    return PyArray_Return(res_a);
}

template<typename T>
void dilate(numpy_aligned_array<T> res, numpy_array<T> array, numpy_aligned_array<T> Bc) {
    const unsigned N = res.size();
    const unsigned N2 = Bc.size();
    numpy_position centre = central_position(Bc);

    typename numpy_array<T>::iterator pos = array.begin();
    for (int i = 0; i != N; ++i, ++pos) {
        if (*pos) {
            typename numpy_aligned_array<T>::iterator startc = Bc.begin();
            for (int j = 0; j != N2; ++j, ++startc) {
                if (*startc) {
                    numpy_position npos = pos.position() + startc.position() - centre;
                    if (res.validposition(npos)) {
                        res.at(npos) = *pos+*startc;
                    }
                }
            }
        } else {
        }
    }
}

PyObject* py_dilate(PyObject* self, PyObject* args, PyObject* kwds) {
    PyArrayObject* array;
    PyArrayObject* Bc;
    static char * kwlist[] = { "array", "Bc", NULL };
    if (!PyArg_ParseTupleAndKeywords(args,kwds,"OO",kwlist,
                    &array,
                    &Bc)) {
        return NULL;
    }
    PyArrayObject* res_a = (PyArrayObject*)PyArray_FromDims(array->nd,array->dimensions,PyArray_TYPE(array));
    if (!res_a) { 
        return NULL;
    }
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
    dilate<type>(numpy_aligned_array<type>(res_a),numpy_array<type>(array),numpy_aligned_array<type>(Bc));\

        HANDLE_INTEGER_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError,"Type not understood.\n");
    }
    return PyArray_Return(res_a);
}

namespace{

PyMethodDef methods[] = {
  {"dilate",(PyCFunction)py_dilate, (METH_VARARGS|METH_KEYWORDS), NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_morph()
  {
    import_array();
    (void)Py_InitModule("_morph", methods);
  }

