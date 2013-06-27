// Copyright (C) 2010-2012 Luis Pedro Coelho <luis@luispedro.org>
//
// License: MIT (see COPYING file)

#include <limits>
#include <memory>
#include <iostream>
#include <assert.h>

#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"

namespace {
using numpy::ndarray_cast;

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _distance (which is dangerous: types are not checked!) or a bug in distance.py.\n";



template<typename BaseType>
inline BaseType square(BaseType x) { return x * x; }

template<typename BaseType>
void dist_transform(BaseType* Df, BaseType* f, const int n, const int stride, double* z, int* v, int* orig, int* ot, const int ostride) {
    const double inf = std::numeric_limits<double>::infinity();
    const double minus_inf = -std::numeric_limits<double>::infinity();
    v[0] = 0;
    z[0] = minus_inf;
    z[1] = inf;
    int k = 0;
    for (int q = 1; q != n; ++q) {
        BaseType s;
        do {
            assert(k >= 0);
            s = ( (f[q*stride] + q*q) - (f[v[k]*stride] + v[k]*v[k])) / 2./ (q-v[k]);
            if (s > z[k]) break;
            --k;
        } while (true);
        ++k;
        v[k] = q;
        z[k] = s;
        z[k+1] = inf;
    }
    k = 0;
    for (int q = 0; q != n; ++q) {
        while (z[k+1] < q) ++k;
        Df[q] = square(q-v[k]) + f[v[k]*stride];
        if (orig) ot[q] = orig[v[k]*ostride];
    }
    for (int q = 0; q != n; ++q) {
        f[q*stride] = Df[q];
        if (orig) orig[q*ostride] = ot[q];
    }
}


PyObject* py_dt(PyObject* self, PyObject* args) {
    PyArrayObject* f;
    PyArrayObject* orig;
    if (!PyArg_ParseTuple(args, "OO", &f, &orig) ||
            !PyArray_Check(f)
            ) {
        PyErr_SetString(PyExc_RuntimeError, "Bad arguments to internal function.");
        return NULL;
    }
    if (PyArray_Check(orig)) {
        if (!PyArray_EquivTypenums(PyArray_TYPE(orig), NPY_INT)) {
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
        Py_INCREF(orig);
    } else {
        orig = 0;
    }
    Py_INCREF(f);
    int* orig_i = (orig? ndarray_cast<int*>(orig) : 0);
    npy_intp* ostrides = (orig ? PyArray_STRIDES(orig) : 0);
    double* z = 0;
    int* v = 0;
    void* Df = 0;
    int* ot = 0;
    const int ndims = PyArray_NDIM(f);
    const int size = PyArray_SIZE(f);
    const npy_intp* strides = PyArray_STRIDES(f);
    npy_intp max_size = 0;
    void* const data = PyArray_DATA(f);

    if (ndims != 2) {
        PyErr_SetString(PyExc_RuntimeError, "_distance only implemented for 2-d arrays.");
        goto exit;
    }
    try {
        for (int k = 0; k != ndims; ++k) {
            npy_intp cur = PyArray_DIM(f, k);
            if (cur > max_size) max_size = cur;
        }
        z = new double[max_size + 1];
        v = new int[max_size];
        Df = operator new(PyArray_ITEMSIZE(f) * max_size);
        ot = (orig ? new int[max_size] : 0);

        for (int k = 0; k != ndims; ++k) {
            const int n = PyArray_DIM(f, k);
            const int outer_n = size/n;
            for (int start = 0; start != outer_n; ++start) {
                int* orig_start = (orig_i ? orig_i + start * ostrides[1-k]/sizeof(int) : 0);
                int ostride = (orig_i ? ostrides[k]/sizeof(int) : 0);
                switch(PyArray_TYPE(f)) {
#define HANDLE(type) { \
                        type* typed_data = static_cast<type*>(data); \
                        const int offset = start*strides[1-k]/sizeof(type); \
                        dist_transform<type>(static_cast<type*>(Df), typed_data + offset, n, strides[k]/sizeof(type), z, v, orig_start, ot, ostride); \
                    }

                    HANDLE_FLOAT_TYPES();
#undef HANDLE
                }
            }
        }
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
    }
    exit:
    delete [] z;
    delete [] v;
    delete [] ot;
    operator delete(Df);
    Py_XDECREF(orig);
    if (PyErr_Occurred()) {
        Py_DECREF(f);
        return NULL;
    }
    return PyArray_Return(f);
}

PyMethodDef methods[] = {
  {"dt", (PyCFunction)py_dt, METH_VARARGS, "Internal function. DO NOT CALL DIRECTLY!"},
  {NULL, NULL,0,NULL},
};

} // namespace

DECLARE_MODULE(_distance)

