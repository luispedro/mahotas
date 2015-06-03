// Copyright (C) 2010-2015  Luis Pedro Coelho <luis@luispedro.org>
//
// License: MIT (see COPYING file)

#include <algorithm>
#include <vector>
#include <cstdio>
#include <limits>
#include <iostream>

#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"


namespace{
using numpy::ndarray_cast;

const char TypeErrorMsg[] = 
    "Type not understood. "
    "This is caused by either a direct call to _bbox (which is dangerous: types are not checked!) or a bug in bbox.py.\n"
    "If you suspect the latter, please report it to the mahotas developpers.";


template<typename T>
void bbox(const numpy::aligned_array<T> array, numpy::index_type* extrema) {
    gil_release nogil;
    const int N = array.size();
    typename numpy::aligned_array<T>::const_iterator pos = array.begin();
    for (int i = 0; i != N; ++i, ++pos) {
        if (*pos) {
            numpy::position where = pos.position();
            for (int j = 0; j != array.ndims(); ++j) {
                extrema[2*j] = std::min<numpy::index_type>(extrema[2*j], where[j]);
                extrema[2*j+1] = std::max<numpy::index_type>(extrema[2*j+1], where[j]+1);
            }
        }
    }
}


template<typename T>
void carray2_bbox(const T* array, const int N0, const int N1, numpy::index_type* extrema) {
    gil_release nogil;
    for (int y = 0; y != N0; ++y) {
        for (int x = 0; x < N1; ++x, ++array)
            if (*array) {
                extrema[0] = std::min<numpy::index_type>(extrema[0], y);
                extrema[1] = std::max<numpy::index_type>(extrema[1], y+1);
                extrema[2] = std::min<numpy::index_type>(extrema[2], x);

                if (static_cast<numpy::index_type>(x+1) < extrema[3]) {
                    // We can skip ahead and go to the end. In a dense array,
                    // we end up skipping most of it.
                    const int step = extrema[3]-x-1;
                    x += step;
                    array += step;
                } else {
                    extrema[3] = x+1;
                }
            }
    }
}

template <typename T>
struct safe_index {
    T* base_;
    int size_;
    safe_index(T* base, const int size)
        :base_(base)
        ,size_(size)
    { }

    safe_index operator + (const int n) {
        return safe_index(base_ + n, size_ - n);
    }
    T& operator * () {
        assert(size_ > 0);
        return *base_;
    }
    T& operator [] (const size_t n) {
        assert(int(n) < size_);
        return base_[n];
    }
};


template<typename T, typename T2>
void carray2_bbox_labeled(const T* array, const int N0, const int N1, T2 extrema) {
    gil_release nogil;
    for (int y = 0; y != N0; ++y) {
        for (int x = 0; x < N1; ++x, ++array) {
            T2 base = extrema + (*array) * 4;
            base[0] = std::min<numpy::index_type>(base[0], y);
            base[1] = std::max<numpy::index_type>(base[1], y+1);

            base[2] = std::min<numpy::index_type>(base[2], x);
            base[3] = std::max<numpy::index_type>(base[3], x+1);
        }
    }
}

template<typename T, typename T2>
void bbox_labeled(const numpy::aligned_array<T> array, T2 extrema) {
    gil_release nogil;
    const int N = array.size();
    const int nd = array.ndim();
    typename numpy::aligned_array<T>::const_iterator pos = array.begin();
    for (int i = 0; i != N; ++i, ++pos) {
        numpy::position where = pos.position();
        T2 base = extrema + (*pos) * 2 * nd;
        for (int j = 0; j != array.ndims(); ++j) {
            base[2*j    ] = std::min<numpy::index_type>(base[2*j  ], where[j]);
            base[2*j + 1] = std::max<numpy::index_type>(base[2*j+1], where[j]+1);
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
    const int nd = PyArray_NDIM(array);
    npy_intp dims[1];
    dims[0] = 2 * nd;
    PyArrayObject* extrema = (PyArrayObject*)PyArray_SimpleNew(1, dims, numpy::index_type_number);

    if (!extrema) return NULL;
    // format for extrema: [ min_0, max_0, min_1, max_1, min_2, max_2, ..., min_k, max_k]
    npy_intp* extrema_v = ndarray_cast<npy_intp*>(extrema);
    for (int j = 0; j != nd; ++j) {
        extrema_v[2*j] = PyArray_DIM(array,j);
        extrema_v[2*j+1] = 0;
    }
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
        if (PyArray_ISCARRAY_RO(array) && nd == 2) { \
            carray2_bbox<type>(ndarray_cast<const type*>(array), PyArray_DIM(array,0), PyArray_DIM(array, 1), extrema_v); \
        } else { \
            bbox<type>(numpy::aligned_array<type>(array), extrema_v); \
        }

        HANDLE_TYPES();
#undef HANDLE
        default:
        Py_DECREF(extrema);
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    if (extrema_v[1] == 0) {
        PyArray_FILLWBYTE(extrema, 0);
    }
    return PyArray_Return(extrema);
}

PyObject* py_bbox_labeled(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* output;
    if (!PyArg_ParseTuple(args,"OO", &array, &output)) return NULL;
    if (!numpy::are_arrays(array, output) ||
            !numpy::is_carray(output)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return 0;
    }
    const int nd = PyArray_NDIM(array);
    const int osize = PyArray_DIM(output, 0);

    if (PyArray_DIM(output, 0) < nd*2) {
        PyErr_SetString(PyExc_RuntimeError, "Output array is not large enough");
        return 0;
    }
    // format for extrema: [ min_0, max_0, min_1, max_1, min_2, max_2, ..., min_k, max_k]
    npy_intp* extrema_v = ndarray_cast<npy_intp*>(output);
    for (int j = 0; j < osize/2; ++j) {
        extrema_v[2*j] = PyArray_DIM(array, j % nd);
        extrema_v[2*j+1] = 0;
    }
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
        if (PyArray_ISCARRAY_RO(array) && nd == 2) { \
            carray2_bbox_labeled<type>(ndarray_cast<const type*>(array), PyArray_DIM(array,0), PyArray_DIM(array, 1), safe_index<npy_intp>(extrema_v, PyArray_DIM(output, 0))); \
        } else { \
            bbox_labeled<type>(numpy::aligned_array<type>(array), safe_index<npy_intp>(extrema_v, PyArray_DIM(output, 0))); \
        }

        HANDLE_INTEGER_TYPES();
#undef HANDLE
#define HANDLE(type) \
        default:
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    // For every possible label, check if it is absent:
    // If so, set all elements to zero
    for (int i = 0; i != osize; i += 2*nd) { // There are osize/(2*nd) possible labels
        if (extrema_v[i + 1] == 0) {
            for (int j = 0; j != 2*nd; ++j ) {
                extrema_v[i + j] = 0;
            }
        }

    }
    Py_INCREF(output);
    return PyArray_Return(output);
}

PyMethodDef methods[] = {
  {"bbox",(PyCFunction)py_bbox, METH_VARARGS, NULL},
  {"bbox_labeled",(PyCFunction)py_bbox_labeled, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
DECLARE_MODULE(_bbox)

