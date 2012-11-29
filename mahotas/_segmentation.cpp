/* Copyright (C) 2012 Luis Pedro Coelho
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
    "This is caused by either a direct call to _segmentation (which is dangerous: types are not checked!) or a bug in segmentation.py.\n";

struct centroid_info {
    centroid_info(float l, float a, float b, float y, float x)
        :l(l)
        ,a(a)
        ,b(b)
        ,y(y)
        ,x(x)
        { }

    centroid_info()
        :l(0)
        ,a(0)
        ,b(0)
        ,y(0)
        ,x(0)
        { }
    float l;
    float a;
    float b;
    float y;
    float x;
};

int slic(const numpy::aligned_array<npy_float32> array, numpy::aligned_array<int> alabels, const int S, const int max_iters=64) {
    assert(alabels.is_carray());
    gil_release no_gil;
    const int Ny = array.dim(0);
    const int Nx = array.dim(1);
    assert(alabels.dim(0) == Ny);
    assert(alabels.dim(1) == Nx);
    const int N = Ny*Nx;
    const float inf = 10e20;
    int* labels = alabels.data();
    std::fill(labels, labels + N, -1);

    std::vector<float> distance;
    distance.resize(N, inf);

    std::vector<centroid_info> centroids;
    std::vector<int> centroid_counts;
    
    for (int y = S/2; y < Ny; y += S) {
        for (int x = S/2; x < Nx; x += S) {
            float l = array.at(y,x,0);
            float a = array.at(y,x,1);
            float b = array.at(y,x,2);
            centroids.push_back(
                centroid_info(l,a,b,y,x));
        }
    }
    centroid_counts.resize(centroids.size());

    for (int i = 0; i < max_iters; ++i) {
        bool changed = false;
        for (unsigned ci = 0; ci < centroids.size(); ++ci) {
            const centroid_info& c = centroids[ci];
            const int start_y = std::max<float>(0.0, c.y - S);
            const int start_x = std::max<float>(0.0, c.x - S);
            const int end_y = std::min<float>(Ny, c.y + S);
            const int end_x = std::min<float>(Nx, c.x + S);
            for (int y = start_y; y != end_y; ++y) {
                for (int x = start_x; x != end_x; ++x) {
                    const int pos = y*Nx + x;
                    float l = array.at(y,x,0);
                    float a = array.at(y,x,1);
                    float b = array.at(y,x,2);
                    const float D = (c.y - y)*(c.y - y) +
                            (c.x - x)*(c.x - x) +
                            (c.l - l)*(c.l - l) +
                            (c.a - a)*(c.a - a) +
                            (c.b - b)*(c.b - b);

                    assert(D < inf);
                    if (D < distance[pos]) {
                        distance[pos] = D;
                        labels[pos] = ci;
                        changed = true;
                    }
                }
            }
        }
        // If nothing changed, we are done
        if (!changed) break;
        std::fill(centroids.begin(), centroids.end(), centroid_info());
        std::fill(centroid_counts.begin(), centroid_counts.end(), 0);
        for (int pos = 0; pos != N; ++pos) {
            const int y = (pos / Nx);
            const int x = (pos % Nx);
            assert(labels[pos] != -1);
            ++centroid_counts[labels[pos]];
            centroid_info& c = centroids[labels[pos]];
            c.l += array.at(y,x,0);
            c.a += array.at(y,x,1);
            c.b += array.at(y,x,2);
            c.y += y;
            c.x += x;
        }
        for (unsigned ci = 0; ci != centroids.size(); ++ci) {
            centroid_info& c = centroids[ci];
            const int cc =  centroid_counts[ci];
            if (cc) {
                c.l /= cc;
                c.a /= cc;
                c.b /= cc;
                c.x /= cc;
                c.y /= cc;
            }
        }
    }
    // Above, we work with labels in 0...[N-1]
    for (unsigned pos = 0; pos != N; ++pos) ++labels[pos];
    return centroids.size();
}

PyObject* py_slic(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* labels;
    int S;
    if (!PyArg_ParseTuple(args,"OOi", &array, &labels, &S)) return NULL;
    if (!numpy::are_arrays(array, labels) ||
        !PyArray_ISCARRAY(array) ||
        !PyArray_ISCARRAY(labels)) {
        PyErr_SetString(PyExc_RuntimeError, "mahotas._segmentation.slic: Need C arrays");
        return NULL;
    }
    if ( !numpy::check_type<npy_float32>(array) ||
        !numpy::check_type<int>(labels)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    if (PyArray_NDIM(array) != 3 ||
        PyArray_NDIM(labels) != 2 ||
        PyArray_DIM(array, 0) != PyArray_DIM(labels, 0) ||
        PyArray_DIM(array, 1) != PyArray_DIM(labels, 1)) {
        PyErr_SetString(PyExc_RuntimeError, "mahotas._segmentation: Unexpected array dimensions");
        return NULL;
    }
    const int n = slic(numpy::aligned_array<npy_float32>(array), numpy::aligned_array<int>(labels), S);
    return PyLong_FromLong(n);
}


PyMethodDef methods[] = {
  {"slic",(PyCFunction)py_slic, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace

DECLARE_MODULE(_segmentation)

