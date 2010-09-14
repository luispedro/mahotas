#include <algorithm>
#include <queue>
#include <vector>
#include <cstdio>
#include <limits>
#include <iostream>

#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _center_of_mass (which is dangerous: types are not checked!) or a bug in center_of_mass.py.\n"
    "If you suspect the latter, please report it to the mahotas developpers.";


template<typename T>
void center_of_mass(const numpy::aligned_array<T> array, npy_double* centers, const npy_int32* labels, double* totals) {
    const unsigned N = array.size();
    const int nd = array.ndims();
    typename numpy::aligned_array<T>::const_iterator pos = array.begin();
    double total = 0;
    for (unsigned i = 0; i != N; ++i, ++pos) {
        double val = *pos;
        if (labels) {
                int label = labels[i];
                totals[label] += val;
                for (int j = 0; j != nd; ++j) {
                    int idx = label*nd + j;
                    centers[idx] += val * pos.index_rev(j);
                }
        } else {
            total += val;
            for (int j = 0; j != nd; ++j) {
                centers[j] += val * pos.index_rev(j);
            }
        }
    }
    if (!labels) *totals = total;
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
        if (!PyArray_Check(labels_obj) ||
            !PyArray_ISCARRAY_RO(labels_obj) ||
            PyArray_TYPE(labels_obj) != NPY_INT32) {
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
        labels = static_cast<const npy_int32*>(PyArray_DATA(labels_obj));
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
        for (int label = 0; label != max_label+1; ++label) totals[label] = 0.0;
    }
    npy_intp dims[1];
    dims[0] = array->nd * (max_label+1);
    PyArrayObject* centers = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!centers) return NULL;
    { // DROP THE GIL
        gil_release nogil;
        npy_double* centers_v = static_cast<npy_double*>(PyArray_DATA(centers));
        for (int j = 0; j != dims[0]; ++j) {
            centers_v[j] = 0;
        }
        switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
            center_of_mass<type>(numpy::aligned_array<type>(array), centers_v, labels, totals); \

            HANDLE_TYPES();
#undef HANDLE
            default: {
                nogil.restore();
                PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
                return NULL;
            }
        }
        int nd = array->nd;
        for (int label = 0; label != (max_label+1); ++label) {
            for (int j = 0; j != nd; ++j) {
                centers_v[label*nd+j] /= totals[label];
            }
            for (int j = 0; j != nd/2; ++j) {
                std::swap(centers_v[label*nd + j], centers_v[(label+1) * nd - j - 1]);
            }
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
extern "C"
void init_center_of_mass()
  {
    import_array();
    (void)Py_InitModule("_center_of_mass", methods);
  }

