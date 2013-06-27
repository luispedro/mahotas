// Copyright (C) 2008-2012 Luis Pedro Coelho <luis@luispedro.org>
// Carnegie Mellon University
// 
// License: MIT (see COPYING file)

#include <vector>
#include <algorithm>
#include <numeric>

#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"

namespace {
using numpy::ndarray_cast;

template<typename BaseType>
void compute_histogram(BaseType* data, int N, unsigned int* histogram) {
    for (int i = 0; i != N; ++i) {
        ++histogram[*data];
        ++data;
    }
}

PyObject* py_histogram(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* histogram;
    if (!PyArg_ParseTuple(args, "OO", &array, &histogram) ||
            !PyArray_Check(array) ||
            !PyArray_Check(histogram) ||
            !PyArray_ISCARRAY(array) ||
            !PyArray_ISCARRAY(histogram) ||
            PyArray_TYPE(histogram) != NPY_UINT
            ) {
        PyErr_SetString(PyExc_RuntimeError, "Bad arguments to internal function.");
        return NULL;
    }
    unsigned int* histogram_data = ndarray_cast<unsigned int*>(histogram);
    const unsigned N = PyArray_SIZE(array);
    switch (PyArray_TYPE(array)) {
        case NPY_UBYTE:
            compute_histogram(ndarray_cast<unsigned char*>(array), N, histogram_data);
            break;
        case NPY_USHORT:
            compute_histogram(ndarray_cast<unsigned short*>(array), N, histogram_data);
            break;
        case NPY_UINT:
            compute_histogram(ndarray_cast<unsigned int*>(array), N, histogram_data);
            break;
        case NPY_ULONG:
            compute_histogram(ndarray_cast<npy_ulong*>(array), N, histogram_data);
            break;
        case NPY_ULONGLONG:
            compute_histogram(ndarray_cast<npy_ulonglong*>(array), N, histogram_data);
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "Cannot handle type.");
            return NULL;
    }
    Py_RETURN_NONE;
}
    

int otsu(const double* hist, const int n) {
// Otsu calculated according to CVonline:
// http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORSE/threshold.pdf
    gil_release nogil;
    std::vector<double> nB, nO;
    if (n <= 1) return 0;
    const double Hsum = std::accumulate(hist + 1, hist + n, double(0));
    if (Hsum == 0) return 0;

    nB.resize(n);
    nB[0] = hist[0];
    for (int i = 1; i != n; ++i) nB[i] = hist[i]+nB[i-1];

    nO.resize(n);
    for (int i = 0; i < n; ++i) nO[i] = nB[n-1]-nB[i];

    double mu_B = 0;
    double mu_O = 0;
    for (int i = 1; i != n; ++i) mu_O += i*hist[i];
    mu_O /= Hsum;

    double best = nB[0]*nO[0]*(mu_B-mu_O)*(mu_B-mu_O);

    int bestT = 0;
    for (int T = 1; T != n; ++T) {
        if (nB[T] == 0) continue;
        if (nO[T] == 0) break;
        mu_B = (mu_B*nB[T-1] + T*hist[T]) / nB[T];
        mu_O = (mu_O*nO[T-1] - T*hist[T]) / nO[T];
        const double sigma_between = nB[T]*nO[T]*(mu_B-mu_O)*(mu_B-mu_O);
        if (sigma_between > best) {
            best = sigma_between;
            bestT = T;
        }
    }
    return bestT;
}


PyObject* py_otsu(PyObject* self, PyObject* args) {
    PyArrayObject* histogram;
    if (!PyArg_ParseTuple(args, "O", &histogram) ||
            !numpy::check_type<double>(histogram)  ||
            !PyArray_ISCARRAY(histogram)
            ) {
        PyErr_SetString(PyExc_RuntimeError, "Bad arguments to internal function.");
        return NULL;
    }
    const double* histogram_data = ndarray_cast<double*>(histogram);
    const unsigned N = PyArray_SIZE(histogram);
    const int res = otsu(histogram_data, N);
    return Py_BuildValue("i", res);
}

PyMethodDef methods[] = {
  {"histogram", (PyCFunction)py_histogram, METH_VARARGS, "Internal function. DO NOT CALL DIRECTLY!"},
  {"otsu", (PyCFunction)py_otsu, METH_VARARGS, "Internal function. DO NOT CALL DIRECTLY!"},
  {NULL, NULL,0,NULL},
};
}

DECLARE_MODULE(_histogram)


