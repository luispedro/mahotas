/* Copyright (C) 2003-2005 Peter J. Verveer
 * Copyright (C) 2011 Luis Pedro Coelho
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. The name of the author may not be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>
#include <math.h>

#include "utils.hpp"
#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace {
const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _interpolate (which is dangerous: types are not checked!) or a bug in interpolate.py.\n";

void init_poles(double pole[2], int& npoles, double& weight, const int order) {
    switch (order) {
    case 2:
        npoles = 1;
        pole[0] = sqrt(8.0) - 3.0;
        break;
    case 3:
        npoles = 1;
        pole[0] = sqrt(3.0) - 2.0;
        break;
    case 4:
        npoles = 2;
        pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
        pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
        break;
    case 5:
        npoles = 2;
        pole[0] = sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5;
        pole[1] = sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5;
        break;
    default:
        throw PythonException(PyExc_RuntimeError, "Order not available (only 2<= order <=5 allowed).");
    }

    weight = 1.0;
    for(int pi = 0; pi < npoles; ++pi) {
        weight *= (1.0 - pole[pi]) * (1.0 - 1.0 / pole[pi]);
    }
}

/* one-dimensional spline filter: */
template<typename FT>
void spline_filter1d(numpy::aligned_array<FT> array, const int order, const int axis) {
    const double log_tolerance = -16.;
    if (axis > array.ndims()) {
        throw PythonException(PyExc_RuntimeError, "Unexpected state.");
    }
    const int len = array.dim(axis);
    const int stride = array.stride(axis);
    if (len <= 1) return;

    int npoles;
    double pole[2];
    double weight;
    init_poles(pole, npoles, weight, order);

    /* these are used in the spline filter calculation below: */

    const int s = array.size();
    typename numpy::aligned_array<FT>::iterator iter = array.begin();
    for (int y = 0; y != s; ++y, ++iter) {
        if (iter.index(axis) != 0) continue;
        FT *line = &*iter;
        /* spline filter: */
        for(int ll = 0; ll < len; ll++) {
            line[stride*ll] *= weight;
        }
        for(int pi = 0; pi < npoles; ++pi) {
            double p = pole[pi];
            int max = (int)ceil(log_tolerance / log(fabs(p)));
            if (max < len) {
                double zn = p;
                double sum = line[0];
                for(int ll = 1; ll < max; ll++) {
                    sum += zn * line[stride*ll];
                    zn *= p;
                }
                line[0] = sum;
            } else {
                double zn = p;
                const double iz = 1.0 / p;
                double z2n = pow(p, (double)(len - 1));
                double sum = line[0] + z2n * line[stride*(len - 1)];
                z2n *= z2n * iz;
                for(int ll = 1; ll <= len - 2; ll++) {
                    sum += (zn + z2n) * line[stride*ll];
                    zn *= p;
                    z2n *= iz;
                }
                line[0] = sum / (1.0 - zn * zn);
            }
            for(int ll = 1; ll < len; ll++)
                line[stride*ll] += p * line[stride*(ll - 1)];
            line[stride*(len-1)] = (p / (p * p - 1.0)) * (line[stride*(len-1)] + p * line[stride*(len-2)]);
            for(int ll = len - 2; ll >= 0; ll--)
                line[stride*ll] = p * (line[stride*(ll + 1)] - line[stride*ll]);
        }
    }
}


PyObject* py_spline_filter1d(PyObject* self, PyObject* args) {

    PyArrayObject* array;
    int order;
    int axis;
    if (!PyArg_ParseTuple(args,"Oii", &array, &order, &axis)) return NULL;
    if (!PyArray_Check(array) || !PyArray_ISCARRAY(array)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    holdref array_hr(array);
#define HANDLE(type) \
    spline_filter1d<type>(numpy::aligned_array<type>(array), order, axis);

    SAFE_SWITCH_ON_FLOAT_TYPES_OF(array, true);
    Py_RETURN_NONE;
}
PyMethodDef methods[] = {
  {"spline_filter1d",(PyCFunction)py_spline_filter1d, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_interpolate()
  {
    import_array();
    (void)Py_InitModule("_interpolate", methods);
  }
