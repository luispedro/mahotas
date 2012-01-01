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

#include <cstdlib>
#include <cmath>
#include <vector>

#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"
#include "_filters.h"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace {
const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _interpolate (which is dangerous: types are not checked!) or a bug in interpolate.py.\n";

template <typename FT>
void init_poles(FT pole[2], int& npoles, FT& weight, const int order) {
    using std::sqrt;
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
    gil_release nogil;
    const FT log_tolerance = -16.;
    if (axis > array.ndims()) {
        throw PythonException(PyExc_RuntimeError, "Unexpected state.");
    }
    const int len = array.dim(axis);
    const int stride = array.stride(axis);
    if (len <= 1) return;

    int npoles;
    FT pole[2];
    FT weight;
    init_poles(pole, npoles, weight, order);

    const int s = array.size();
    typename numpy::aligned_array<FT>::iterator iter = array.begin();
    for (int y = 0; y != s; ++y, ++iter) {
        if (iter.index(axis) != 0) continue;
        FT *line = &*iter;
        for(int ll = 0; ll < len; ll++) {
            line[stride*ll] *= weight;
        }
        for(int pi = 0; pi < npoles; ++pi) {
            FT p = pole[pi];
            int max = (int)std::ceil(log_tolerance / log(fabs(p)));
            if (max < len) {
                FT zn = p;
                FT sum = line[0];
                for(int ll = 1; ll < max; ll++) {
                    sum += zn * line[stride*ll];
                    zn *= p;
                }
                line[0] = sum;
            } else {
                FT zn = p;
                const FT iz = 1.0 / p;
                FT z2n = pow(p, (FT)(len - 1));
                FT sum = line[0] + z2n * line[stride*(len - 1)];
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


template <typename FT>
void spline_coefficients(FT x, const int order, std::vector<FT>& result)
{
    const FT start = floor(x + 0.5*(order & 1)) - order / 2;

    for(int hh = 0; hh <= order; hh++)  {
        FT y = fabs(start - x + hh);

        switch(order) {
        case 1:
            result[hh] = y > 1.0 ? 0.0 : 1.0 - y;
            break;
        case 2:
            if (y < 0.5) {
                result[hh] = 0.75 - y * y;
            } else if (y < 1.5) {
                y = 1.5 - y;
                result[hh] = 0.5 * y * y;
            } else {
                result[hh] = 0.0;
            }
            break;
        case 3:
            if (y < 1.0) {
                result[hh] =
                    (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
            } else if (y < 2.0) {
                y = 2.0 - y;
                result[hh] = y * y * y / 6.0;
            } else {
                result[hh] = 0.0;
            }
            break;
        case 4:
            if (y < 0.5) {
                y *= y;
                result[hh] = y * (y * 0.25 - 0.625) + 115.0 / 192.0;
            } else if (y < 1.5) {
                result[hh] = y * (y * (y * (5.0 / 6.0 - y / 6.0) - 1.25) + 5.0 / 24.0) + 55.0 / 96.0;
            } else if (y < 2.5) {
                y -= 2.5;
                y *= y;
                result[hh] = y * y / 24.0;
            } else {
                result[hh] = 0.0;
            }
            break;
        case 5:
            if (y < 1.0) {
                const FT f = y * y;
                result[hh] = f * (f * (0.25 - y / 12.0) - 0.5) + 0.55;
            } else if (y < 2.0) {
                result[hh] = y * (y * (y * (y * (y / 24.0 - 0.375) + 1.25) -  1.75) + 0.625) + 0.425;
            } else if (y < 3.0) {
                const FT f = 3.0 - y;
                y = f * f;
                result[hh] = f * y * y / 120.0;
            } else {
                result[hh] = 0.0;
            }
            break;
        }
    }
}

inline
int int_pow(const int x, const int p) {
    int r = 1;
    for (int i = 0; i != p; ++i) r *= x;
    return r;
}

template <typename FT>
void zoom_shift(numpy::aligned_array<FT> array, PyArrayObject* zoom_ar,
                                 PyArrayObject* shift_ar, numpy::aligned_array<FT> output,
                                 int order, int mode, FT cval) {
    gil_release nogil;
    typename numpy::aligned_array<FT>::iterator io = output.begin();
    FT *zooms = zoom_ar ? (FT*)PyArray_DATA(zoom_ar) : NULL;
    FT *shifts = shift_ar ? (FT*)PyArray_DATA(shift_ar) : NULL;
    const int rank = array.ndims();

    std::vector< std::vector<bool> > zeros;
    /* if the mode is 'constant' we need some temps later: */
    if (mode == EXTEND_CONSTANT) {
        for(int r = 0; r < rank; r++) {
            zeros.push_back( std::vector<bool>(output.dim(r)) );
        }
    }

    /* store offsets, along each axis: */
    std::vector< std::vector<npy_intp> > offsets;
    /* store spline coefficients, along each axis: */
    std::vector< std::vector< std::vector<FT> > > splvals;
    /* store offsets at all edges: */
    std::vector< std::vector< std::vector<npy_intp> > > edge_offsets;
    for(int r = 0; r < rank; ++r) {
        offsets.push_back( std::vector<npy_intp>(output.dim(r)) );
        splvals.push_back( std::vector< std::vector<FT> >(output.dim(r)) );
        edge_offsets.push_back( std::vector< std::vector<npy_intp> >(output.dim(r)) );
    }

    /* precalculate offsets, and offsets at the edge: */
    for(int r = 0; r < rank; r++) {
        double shift = 0.0, zoom = 0.0;
        if (shifts) shift = shifts[r];
        if (zooms) zoom = zooms[r];
        for(int kk = 0; kk < output.dim(r); ++kk) {
            FT cc = kk;
            if (shifts) cc += shift;
            if (zooms) cc *= zoom;
            cc = fix_offset(ExtendMode(mode), cc, array.dim(r), -1);
            if (cc != -1) {
                const int start = int(floor(cc + 0.5*(order & 1)) - order / 2);
                offsets[r][kk] = array.stride(r) * start;
                if (start < 0 || start + order >= array.dim(r)) {
                    edge_offsets[r][kk].resize(order + 1);
                    for(int hh = 0; hh <= order; hh++) {
                        int idx = start + hh;
                         int len = array.dim(r);
                        if (len <= 1) {
                            idx = 0;
                        } else {
                            int s2 = 2 * len - 2;
                            if (idx < 0) {
                                idx = s2 * (int)(-idx / s2) + idx;
                                idx = idx <= 1 - len ? idx + s2 : -idx;
                            } else if (idx >= len) {
                                idx -= s2 * (int)(idx / s2);
                                if (idx >= len)
                                    idx = s2 - idx;
                            }
                        }
                        edge_offsets[r][kk][hh] = array.stride(r) * (idx - start);
                    }
                }
                if (order > 0) {

                    splvals[r][kk].resize(order + 1);
                    spline_coefficients(cc, order, splvals[r][kk]);
                }
            } else {
                zeros[r][kk] = true;
            }
        }
    }

    const int filter_size = int_pow(order + 1, rank);
    std::vector<npy_intp> idxs(filter_size);
    std::vector<npy_intp> fcoordinates(rank * filter_size);
    std::vector<npy_intp> foffsets(filter_size);

    std::vector<npy_intp> ftmp(rank);
    int off = 0;
    for(int hh = 0; hh < filter_size; hh++) {
        for(int r = 0; r < rank; r++)
            fcoordinates[r + hh * rank] = ftmp[r];
        foffsets[hh] = off;
        for(int r = rank - 1; r >= 0; r--) {
            if (ftmp[r] < order) {
                ftmp[r]++;
                off += array.stride(r);
                break;
            } else {
                ftmp[r] = 0;
                off -= array.stride(r) * order;
            }
        }
    }
    const npy_intp size = output.size();
    for(int i = 0; i < size; ++i, ++io) {
        int oo = 0;
        bool on_edge = false;
        bool zero = false;
        for(int r = 0; r < rank; r++) {
            if (zeros.size() && zeros[r][io.index(r)]) {
                /* we use constant border condition */
                *io = cval;
                zero = true;
                break;
            }
            oo += offsets[r][io.index(r)];
            if (edge_offsets[r][io.index(r)].size()) on_edge = true;
        }
        if (zero) continue;
        std::vector<npy_intp>::const_iterator ff = fcoordinates.begin();
        for(int fi = 0; fi < filter_size; fi++) {
            int idx = 0;
            if (on_edge) {
                    /* use precalculated edge offsets: */
                for(int r = 0; r < rank; r++) {
                    if (edge_offsets[r][io.index(r)].size())
                        idx += edge_offsets[r][io.index(r)][ff[r]];
                    else
                        idx += ff[r] * array.stride(r);
                }
                idx += oo;
            } else {
                /* use normal offsets: */
                idx += oo + foffsets[fi];
            }
            idxs[fi] = idx;
            ff += rank;
        }
        ff = fcoordinates.begin();
        FT t = 0.0;
        for(int fi = 0; fi < filter_size; fi++) {
            double coeff = array.data()[idxs[fi]];
            /* calculate interpolated value: */
            for(int r = 0; r < rank; r++)
                if (order > 0)
                    coeff *= splvals[r][io.index(r)][ff[r]];
            t += coeff;
            ff += rank;
        }
        *io = t;
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
#undef HANDLE

    Py_RETURN_NONE;
}


PyObject* py_zoom_shift(PyObject* self, PyObject* args) {

    PyArrayObject* array;
    PyArrayObject* zooms;
    PyArrayObject* shifts;
    PyArrayObject* output;
    int order;
    int mode;
    double cval;
    if (!PyArg_ParseTuple(args,"OOOOiif", &array, &zooms, &shifts, &output, &order, &mode, &cval)) return NULL;
    if (!PyArray_Check(array) || !PyArray_ISCARRAY(array) ||
        !PyArray_Check(output) || !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    if (!PyArray_Check(zooms)) {
        zooms = 0;
    } else if (!PyArray_ISCARRAY(zooms)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }

    if (!PyArray_Check(shifts)) {
        shifts = 0;
    } else if (!PyArray_ISCARRAY(shifts)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    holdref array_hr(array);
    holdref zoom_hr(zooms);
    holdref shifts_hr(shifts);
    holdref output_hr(output);
#define HANDLE(type) \
    zoom_shift<type>(numpy::aligned_array<type>(array), zooms, shifts, numpy::aligned_array<type>(output), order, mode, type(cval));
    SAFE_SWITCH_ON_FLOAT_TYPES_OF(array, true);
#undef HANDLE

    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
  {"spline_filter1d",(PyCFunction)py_spline_filter1d, METH_VARARGS, NULL},
  {"zoom_shift",(PyCFunction)py_zoom_shift, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_interpolate()
  {
    import_array();
    (void)Py_InitModule("_interpolate", methods);
  }
