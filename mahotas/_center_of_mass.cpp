// Copyright (C) 2008-2012  Luis Pedro Coelho <luis@luispedro.org>
//
// License: MIT (see COPYING file)

#include <algorithm>
#include <vector>
#include <limits>

#include "numpypp/array.hpp"
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"

namespace{
using numpy::ndarray_cast;

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _center_of_mass (which is dangerous: types are not checked!) or a bug in center_of_mass.py.\n"
    "If you suspect the latter, please report it to the mahotas developpers.";


template<typename T>
void center_of_mass(const numpy::aligned_array<T> array, npy_double* centers, const npy_int32* labels, double* totals) {
    const unsigned N = array.size();
    const int nd = array.ndims();
    typename numpy::aligned_array<T>::const_iterator pos = array.begin();
    for (unsigned i = 0; i != N; ++i, ++pos) {
        const double val = *pos;
        const int label = (labels ? labels[i] : 0);
        totals[label] += val;
        npy_double* centers_label = centers + label*nd;
        for (int j = 0; j != nd; ++j) {
            centers_label[j] += val * pos.index_rev(j);
        }
    }
}

PyObject* py_center_of_mass(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyObject* labels_obj;
    const npy_int32* labels = 0;
    double total_sum = 0.0;
    double * totals = &total_sum;
    int max_label = 0;
    if (!PyArg_ParseTuple(args,"OO", &array, &labels_obj)) return NULL;
    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    if (labels_obj != Py_None) {
        PyArrayObject* labels_arr = (PyArrayObject*)(labels_obj);
        if (!PyArray_Check(labels_obj) ||
            !PyArray_ISCARRAY_RO(labels_arr) ||
            !numpy::check_type<npy_int32>(labels_arr)) {
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
        labels = ndarray_cast<const npy_int32*>(labels_arr);
    }
    holdref labels_obj_hr(labels_obj);
    if (labels) {
        const int N = PyArray_SIZE(array);
        for (int i = 0; i != N; ++i) {
            if (labels[i] < 0) {
                PyErr_SetString(PyExc_RuntimeError, "Labels array cannot be negative.");
                return NULL;
            }
            if (labels[i] > max_label) max_label = labels[i];
        }
        totals = new(std::nothrow) double[max_label+1];
        if (!totals) {
            PyErr_NoMemory();
            return NULL;
        }
        std::fill(totals, totals + max_label + 1, 0.0);
    }
    npy_intp dims[1];
    dims[0] = PyArray_NDIM(array) * (max_label+1);
    PyArrayObject* centers = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!centers) return NULL;
    { // DROP THE GIL
        gil_release nogil;
        npy_double* centers_v = ndarray_cast<npy_double*>(centers);
        std::fill(centers_v, centers_v + dims[0], 0);
        switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
            center_of_mass<type>(numpy::aligned_array<type>(array), centers_v, labels, totals); \

            HANDLE_TYPES();
#undef HANDLE
            default: {
                if (labels) delete [] totals;
                nogil.restore();
                PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
                return NULL;
            }
        }
        const int nd = PyArray_NDIM(array);
        for (int label = 0; label != (max_label+1); ++label) {
            for (int j = 0; j != nd; ++j) {
                centers_v[label*nd+j] /= totals[label];
            }
            std::reverse(centers_v + label*nd, centers_v + (label+1)*nd);
        }
        if (labels) delete [] totals;
    }
    return PyArray_Return(centers);
}


PyMethodDef methods[] = {
  {"center_of_mass",(PyCFunction)py_center_of_mass, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace

DECLARE_MODULE(_center_of_mass)
