
// Copyright (C) 2010-2012 Luis Pedro Coelho <luis@luispedro.org>
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published
// by the Free Software Foundation; either version 2 of the License,
// or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
// 02110-1301, USA.



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
    "This is caused by either a direct call to _convolve (which is dangerous: types are not checked!) or a bug in convolve.py.\n";
const char OutputErrorMsg[] =
    "Output type is not valid. "
    "This is caused by either a direct call to _convolve (which is dangerous: types are not checked!) or a bug in convolve.py.\n";


template<typename T>
void convolve(numpy::aligned_array<T> array, numpy::aligned_array<T> filter, numpy::aligned_array<T> result, int mode) {
    gil_release nogil;
    const int N = array.size();
    typename numpy::aligned_array<T>::iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), filter.raw_array(), ExtendMode(mode), true);
    const int N2 = fiter.size();
    T* out = result.data();

    for (int i = 0; i != N; ++i, fiter.iterate_with(iter), ++iter, ++out) {
        // The reasons for using double instead of T:
        //   (1) it is slightly faster (10%)
        //   (2) it handles over/underflow better
        //   (3) scipy.ndimage.convolve does it
        // 
        // Alternatively, we could have written:
        // T cur = T();
        // 
        // and removed the double cast in double(val)*fiter[j] below.
        double cur = 0.;
        for (int j = 0; j != N2; ++j) {
            T val;
            if (fiter.retrieve(iter, j, val)) {
                cur += double(val)*fiter[j];
            }
        }
        *out = T(cur);
    }
}


PyObject* py_convolve(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    PyArrayObject* output;
    int mode;
    if (!PyArg_ParseTuple(args,"OOOi", &array, &filter, &output, &mode)) return NULL;
    if (!PyArray_Check(array) || !PyArray_Check(filter) || PyArray_TYPE(array) != PyArray_TYPE(filter) || PyArray_NDIM(array) != PyArray_NDIM(filter)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }

    if (reinterpret_cast<PyObject*>(output) == Py_None) {
        output = reinterpret_cast<PyArrayObject*>(
                PyArray_EMPTY(PyArray_NDIM(array), PyArray_DIMS(array), PyArray_TYPE(array), 0));
        if (!output) return NULL;
    } else {
        if (!PyArray_Check(output) ||
            PyArray_NDIM(output) != PyArray_NDIM(array) ||
            PyArray_TYPE(output) != PyArray_TYPE(array) ||
            !PyArray_ISCARRAY(output)) {
            PyErr_SetString(PyExc_RuntimeError, OutputErrorMsg);
            return NULL;
        }
        for (int d = 0; d != PyArray_NDIM(array); ++d) {
            if (PyArray_DIM(array, d) != PyArray_DIM(output, d)) {
                PyErr_SetString(PyExc_RuntimeError, OutputErrorMsg);
                return NULL;
            }
        }
        Py_INCREF(output);
    }

#define HANDLE(type) \
    convolve<type>(numpy::aligned_array<type>(array), numpy::aligned_array<type>(filter), numpy::aligned_array<type>(output), mode);
    SAFE_SWITCH_ON_TYPES_OF(array, true)
#undef HANDLE
    return PyArray_Return(output);
}

template<typename T>
void rank_filter(numpy::aligned_array<T> res, numpy::aligned_array<T> array, numpy::aligned_array<T> Bc, const int rank, const int mode) {
    gil_release nogil;
    const int N = res.size();
    typename numpy::aligned_array<T>::iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), Bc.raw_array(), ExtendMode(mode), true);
    const int N2 = fiter.size();
    if (rank < 0 || rank >= N2) {
        return;
    }
    // T* is a fine iterator type.
    T* rpos = res.data();
    T* neighbours = new T[N2];

    for (int i = 0; i != N; ++i, ++rpos, fiter.iterate_both(iter)) {
        int n = 0;
        for (int j = 0; j != N2; ++j) {
            T val;
            if (fiter.retrieve(iter, j, val)) neighbours[n++] = val;
        }
        int currank = rank;
        if (n != N2) {
            currank = int(n * rank/float(N2));
        }
        std::nth_element(neighbours, neighbours + currank, neighbours + n);
        *rpos = neighbours[rank];
    }
    delete [] neighbours;
}
PyObject* py_rank_filter(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* Bc;
    int rank;
    int mode;
    PyArrayObject* output;
    if (!PyArg_ParseTuple(args, "OOOii", &array, &Bc, &output, &rank, &mode) ||
        !PyArray_Check(array) || !PyArray_Check(Bc) || !PyArray_Check(output) ||
        !PyArray_EquivTypenums(PyArray_TYPE(array), PyArray_TYPE(Bc)) ||
        !PyArray_EquivTypenums(PyArray_TYPE(array), PyArray_TYPE(output)) ||
        !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    holdref r(output);

#define HANDLE(type) \
        rank_filter<type>(numpy::aligned_array<type>(output), numpy::aligned_array<type>(array), numpy::aligned_array<type>(Bc), rank, mode);
    SAFE_SWITCH_ON_TYPES_OF(array,true)
#undef HANDLE

    Py_INCREF(output);
    return PyArray_Return(output);
}

template <typename T>
void template_match(numpy::aligned_array<T> res, numpy::aligned_array<T> f, numpy::aligned_array<T> t, int mode) {
    gil_release nogil;
    const int N = res.size();
    typename numpy::aligned_array<T>::iterator iter = f.begin();
    filter_iterator<T> fiter(f.raw_array(), t.raw_array(), ExtendMode(mode), false);
    const int N2 = fiter.size();
    // T* is a fine iterator type.
    T* rpos = res.data();

    for (int i = 0; i != N; ++i, ++rpos, fiter.iterate_both(iter)) {
        T diff2 = T(0);
        for (int j = 0; j != N2; ++j) {
            T val;
            if (fiter.retrieve(iter, j, val)) {
                const T tj = fiter[j];
                const T delta = (val > tj ? val - tj : tj - val);
                diff2 += delta*delta;
            }
        }
        *rpos = diff2;
    }
}

PyObject* py_template_match(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* template_;
    int mode;
    PyArrayObject* output;
    if (!PyArg_ParseTuple(args, "OOOi", &array, &template_, &output, &mode) ||
        !PyArray_Check(array) || !PyArray_Check(output) || !PyArray_Check(template_) ||
        PyArray_TYPE(array) != PyArray_TYPE(output) ||
        PyArray_TYPE(template_) != PyArray_TYPE(array) ||
        !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    holdref r(output);

#define HANDLE(type) \
        template_match<type>(numpy::aligned_array<type>(output), numpy::aligned_array<type>(array), numpy::aligned_array<type>(template_), mode);
    SAFE_SWITCH_ON_TYPES_OF(array,true)
#undef HANDLE

    Py_INCREF(output);
    return PyArray_Return(output);
}

PyMethodDef methods[] = {
  {"convolve",(PyCFunction)py_convolve, METH_VARARGS, NULL},
  {"rank_filter",(PyCFunction)py_rank_filter, METH_VARARGS, NULL},
  {"template_match",(PyCFunction)py_template_match, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_convolve()
  {
    import_array();
    (void)Py_InitModule("_convolve", methods);
  }

