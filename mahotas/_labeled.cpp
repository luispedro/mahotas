/* Copyright (C) 2010-2014  Luis Pedro Coelho <luis@luispedro.org>
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
#include <map>
#include <queue>
#include <functional>

#include "numpypp/array.hpp"
#include "numpypp/numpy.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"
#include "_filters.h"
#include "debug.h"

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _labeled (which is dangerous: types are not checked!) or a bug in labeled.py.\n";


// This is a standard union-find structure
template<typename It>
int find(It data, int i) {
    if (data[i] == i) return i;
    int j = find(data, data[i]);
    data[i] = j;
    return j;
}
template<typename It>
void compress(It data, int i) {
    find(data,i);
}

template<typename It>
void join(It data, int i, int j) {
    i = find(data, i);
    j = find(data, j);
    assert(i >= 0);
    assert(j >= 0);
    data[i] = j;
}

int label(numpy::aligned_array<int> labeled, const numpy::aligned_array<int> Bc) {
    gil_release nogil;
    const int N = labeled.size();
    get_pointer_type<int>::ptr data = as_checked_ptr(labeled.data(), N);
    for (int i = 0; i != N; ++i) {
        data[i] = (data[i] ? i : -1);
    }
    numpy::aligned_array<int>::iterator iter = labeled.begin();
    filter_iterator<int> filter(labeled.raw_array(), Bc.raw_array());
    const int N2 = filter.size();
    for (int i = 0; i != N; ++i, filter.iterate_both(iter)) {
        if (*iter != -1) {
            for (int j = 0; j != N2; ++j) {
                int arr_val = false;
                filter.retrieve(iter, j, arr_val);
                if (arr_val != -1) {
                    join(data, i, arr_val);
                }
            }
        }
    }
    for (int i = 0; i != N; ++i) {
        if (data[i] != -1) compress(data, i);
    }
    int next = 1;
    std::map<int, int> seen;
    seen[-1] = 0;
    for (int i = 0; i != N; ++i) {
        const int val = data[i];
        std::map<int, int>::iterator where = seen.find(val);
        if (where == seen.end()) {
            data[i] = next;
            seen[val] = next;
            ++next;
        } else {
            data[i] = where->second;
        }
    }
    return (next - 1);
}

int relabel(numpy::aligned_array<int> labeled) {
    gil_release nogil;
    const int N = labeled.size();
    int* data = labeled.data();
    int next = 1;
    std::map<int, int> seen;
    seen[0] = 0;
    for (int i = 0; i != N; ++i) {
        const int val = data[i];
        std::map<int, int>::iterator where = seen.find(val);
        if (where == seen.end()) {
            data[i] = next;
            seen[val] = next;
            ++next;
        } else {
            data[i] = where->second;
        }
    }
    return (next - 1);
}

bool is_same_labeling(numpy::aligned_array<int> labeled0, numpy::aligned_array<int> labeled1) {
    gil_release nogil;
    std::map<int,int> index;
    std::map<int,int> rindex;
    index[0] = 0;
    rindex[0] = 0;
    const int N = labeled0.size();
    assert(labeled1.size() == N);
    const int* a = labeled0.data();
    const int* b = labeled1.data();
    for (int p = 0; p < N; ++p, ++a, ++b) {
        std::map<int,int>::const_iterator va =  index.insert(std::make_pair(*a, *b)).first;
        std::map<int,int>::const_iterator vb = rindex.insert(std::make_pair(*b, *a)).first;

        if (va->second != *b || vb->second != *a) {
            return false;
        }
    }
    return true;
}

void remove_regions(numpy::aligned_array<int> labeled, numpy::aligned_array<int> regions) {
    gil_release nogil;
    const int N = labeled.size();
    int* data = labeled.data();

    const int* const r_start = regions.data();
    const int* const r_end = regions.data() + regions.size();
    for (int i = 0; i != N; ++i) {
        if (data[i] && std::binary_search(r_start, r_end, data[i])) data[i] = 0;
    }
}



template<typename T>
void borders(const numpy::aligned_array<T> array, const numpy::aligned_array<T> filter, numpy::aligned_array<bool> result, int mode) {
    gil_release nogil;
    const int N = array.size();
    typename numpy::aligned_array<T>::const_iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), filter.raw_array(), ExtendMode(mode), true);
    const int N2 = fiter.size();
    bool* out = result.data();

    for (int i = 0; i != N; ++i, fiter.iterate_both(iter), ++out) {
        const T cur = *iter;
        for (int j = 0; j != N2; ++j) {
            T val ;
            if (fiter.retrieve(iter, j, val) && (val != cur)) {
                *out = true;
                break; // goto next i
            }
        }
    }
}

template<typename T>
bool border(const numpy::aligned_array<T> array, const numpy::aligned_array<T> filter, numpy::aligned_array<bool> result, T i, T j) {
    gil_release nogil;
    const int N = array.size();
    typename numpy::aligned_array<T>::const_iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), filter.raw_array(), ExtendConstant, true);
    const int N2 = fiter.size();
    bool* out = result.data();
    bool any = false;

    for (int ii = 0; ii != N; ++ii, fiter.iterate_both(iter), ++out) {
        const T cur = *iter;
        T other;
        if (cur == i) other = j;
        else if (cur == j) other = i;
        else continue;
        for (int j = 0; j != N2; ++j) {
            T val ;
            if (fiter.retrieve(iter, j, val) && (val == other)) {
                *out = true;
                any = true;
            }
        }
    }
    return any;
}

template <typename T, typename F>
void labeled_foldl(const numpy::aligned_array<T> array, const numpy::aligned_array<int> labeled, T* result, const int maxlabel, const T start, F f) {
    gil_release nogil;
    typename numpy::aligned_array<T>::const_iterator iterator = array.begin();
    numpy::aligned_array<int>::const_iterator literator = labeled.begin();
    const int N = array.size();
    std::fill(result, result + maxlabel, start);
    for (int i = 0; i != N; ++i, ++iterator, ++literator) {
        if ((*literator >= 0) && (*literator < maxlabel)) {
            result[*literator] = f(*iterator, result[*literator]);
        }
    }
}
// In certain versions of g++, in certain environments,
// multiple versions of std::min & std::max are in scope and the compiler is
// unable to resolve them. Therefore, based on
// http://www.cplusplus.com/reference/algorithm/min/ &
// http://www.cplusplus.com/reference/algorithm/max/ I implemented equivalents
// here:

template <class T>
const T& std_like_min(const T& a, const T& b) {
      return !(b<a)?a:b;
}
template <class T>
const T& std_like_max(const T& a, const T& b) {
      return (a<b)?b:a;
}

template <typename T>
void labeled_sum(const numpy::aligned_array<T> array, const numpy::aligned_array<int> labeled, T* result, const int maxlabel) {
    labeled_foldl(array, labeled, result, maxlabel, T(), std::plus<T>());
}
template <>
void labeled_sum<bool>(const numpy::aligned_array<bool> array, const numpy::aligned_array<int> labeled, bool* result, const int maxlabel) {
    labeled_foldl(array, labeled, result, maxlabel, false, std::logical_or<bool>());
}

template <typename T>
void labeled_max(const numpy::aligned_array<T> array, const numpy::aligned_array<int> labeled, T* result, const int maxlabel) {
    labeled_foldl(array, labeled, result, maxlabel,std::numeric_limits<T>::min(), std_like_max<T>);
}

template <typename T>
void labeled_min(const numpy::aligned_array<T> array, const numpy::aligned_array<int> labeled, T* result, const int maxlabel) {
    labeled_foldl(array, labeled, result, maxlabel,std::numeric_limits<T>::max(), std_like_min<T>);
}


PyObject* py_label(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    if (!PyArg_ParseTuple(args,"OO", &array, &filter)) return NULL;
    if (!numpy::are_arrays(array, filter) ||
        !numpy::equiv_typenums(array, filter) ||
        !numpy::check_type<int>(array) ||
        !PyArray_ISCARRAY(array)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    int n = label(numpy::aligned_array<int>(array), numpy::aligned_array<int>(filter));
    return PyLong_FromLong(n);
}

PyObject* py_relabel(PyObject* self, PyObject* args) {
    PyArrayObject* labeled;
    if (!PyArg_ParseTuple(args,"O", &labeled)) return NULL;
    if (!numpy::are_arrays(labeled) ||
        !numpy::check_type<int>(labeled) ||
        !PyArray_ISCARRAY(labeled)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    int n = relabel(numpy::aligned_array<int>(labeled));
    return PyLong_FromLong(n);
}

PyObject* py_is_same_labeling(PyObject* self, PyObject* args) {
    PyArrayObject* labeled0;
    PyArrayObject* labeled1;
    if (!PyArg_ParseTuple(args,"OO", &labeled0, &labeled1)) return NULL;
    if (!numpy::are_arrays(labeled0, labeled1) ||
        !numpy::check_type<int>(labeled0) ||
        !numpy::check_type<int>(labeled1) ||
        !PyArray_ISCARRAY(labeled0) ||
        !PyArray_ISCARRAY(labeled1)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    bool same = is_same_labeling(numpy::aligned_array<int>(labeled0), numpy::aligned_array<int>(labeled1));
    return PyBool_FromLong(same);
}

PyObject* py_remove_regions(PyObject* self, PyObject* args) {
    PyArrayObject* labeled;
    PyArrayObject* regions;
    if (!PyArg_ParseTuple(args,"OO", &labeled, &regions)) return NULL;
    if (!numpy::are_arrays(labeled, regions) ||
        !numpy::check_type<int>(labeled) ||
        !numpy::check_type<int>(regions) ||
        !PyArray_ISCARRAY(labeled) ||
        !PyArray_ISCARRAY(regions)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    remove_regions(numpy::aligned_array<int>(labeled), numpy::aligned_array<int>(regions));
    return PyLong_FromLong(0);
}

PyObject* py_borders(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    PyArrayObject* output;
    int mode;
    if (!PyArg_ParseTuple(args,"OOOi", &array, &filter, &output, &mode)) return NULL;
    if (!numpy::are_arrays(array, filter, output) ||
        !numpy::equiv_typenums(array, filter) ||
        !numpy::check_type<bool>(output) ||
        !numpy::same_shape(array, output) ||
        !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    holdref ro(output);

#define HANDLE(type) \
    borders<type>( \
                numpy::aligned_array<type>(array), \
                numpy::aligned_array<type>(filter), \
                numpy::aligned_array<bool>(output), \
                mode);
    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(output);
    return PyArray_Return(output);
}

PyObject* py_border(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    PyArrayObject* output;
    int i;
    int j;
    int always_return;
    if (!PyArg_ParseTuple(args,"OOOiii", &array, &filter, &output, &i, &j, &always_return)) return NULL;
    if (!numpy::are_arrays(array, filter, output) ||
        !numpy::equiv_typenums(array, filter) ||
        !numpy::check_type<bool>(output) ||
        !numpy::same_shape(array, output) ||
        !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    holdref ro(output);

    bool has_any;
#define HANDLE(type) \
    has_any = !!border<type>( \
                numpy::aligned_array<type>(array), \
                numpy::aligned_array<type>(filter), \
                numpy::aligned_array<bool>(output), \
                static_cast<type>(i), \
                static_cast<type>(j));
    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE
    if (always_return || has_any) {
        Py_INCREF(output);
        return PyArray_Return(output);
    }

    Py_RETURN_NONE;
}

PyObject* py_labeled_sum(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* labeled;
    PyArrayObject* output;
    if (!PyArg_ParseTuple(args,"OOO", &array, &labeled, &output)) return NULL;
    if (!numpy::are_arrays(array, labeled, output) ||
        !numpy::same_shape(array, labeled) ||
        !numpy::equiv_typenums(array, output) ||
        !numpy::check_type<int>(labeled) ||
        !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    const int maxi = PyArray_DIM(output, 0);

#define HANDLE(type) \
    { \
        type* odata = numpy::ndarray_cast<type*>(output); \
        labeled_sum<type>( \
                numpy::aligned_array<type>(array), \
                numpy::aligned_array<int>(labeled), \
                odata, \
                maxi); \
    }
    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE

    Py_RETURN_NONE;
}
PyObject* py_labeled_max_min(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* labeled;
    PyArrayObject* output;
    int is_max;
    if (!PyArg_ParseTuple(args,"OOOi", &array, &labeled, &output, &is_max)) return NULL;
    if (!numpy::are_arrays(array, labeled, output) ||
        !numpy::same_shape(array, labeled) ||
        !numpy::equiv_typenums(array, output) ||
        !numpy::check_type<int>(labeled) ||
        !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    const int maxi = PyArray_DIM(output, 0);

#define HANDLE(type) \
    { \
        type* odata = numpy::ndarray_cast<type*>(output); \
        if (is_max) { \
            labeled_max<type>( \
                numpy::aligned_array<type>(array), \
                numpy::aligned_array<int>(labeled), \
                odata, \
                maxi); \
        } else { \
            labeled_min<type>( \
                numpy::aligned_array<type>(array), \
                numpy::aligned_array<int>(labeled), \
                odata, \
                maxi); \
        } \
    }
    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE

    Py_RETURN_NONE;
}



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

bool fetch_neighbour(int n, int y, int x, int& ny, int& nx, const int Ny, const int Nx) {
    ny = y;
    nx = x;
    switch(n) {
        case 0:
            ++nx;
            if (nx < Nx) return true;
            return false;
        case 1:
            --nx;
            if (nx >= 0) return true;
            return false;
        case 2:
            ++ny;
            if (ny < Ny) return true;
            return false;
        case 3:
            --ny;
            if (ny >= 0) return true;
            return false;
        default:
            assert(0);
            return false; // Get rid of "control reaches end of unreachable function" warning
    }
}

struct SlicPoint {
    SlicPoint(int y, int x, int ci, int cost)
        :y(y)
        ,x(x)
        ,ci(ci)
        ,cost(cost)
        { }

    bool operator < (const SlicPoint& other) const { return this->cost > other.cost; }

    int y;
    int x;
    int ci;
    int cost;
};

int diff2(int a, int b) { return (a-b)*(a-b); }

int slic(const numpy::aligned_array<npy_float32> array, numpy::aligned_array<int> alabels, const int S, const float m, const int max_iters) {
    assert(alabels.is_carray());
    gil_release no_gil;
    const int Ny = array.dim(0);
    const int Nx = array.dim(1);
    assert(alabels.dim(0) == Ny);
    assert(alabels.dim(1) == Nx);
    const int N = Ny*Nx;
    const float inf = 10e20;
    // m²/S²
    const float m2S2 = float(m*m)/float(S*S);
    int* labels = alabels.data();
    std::fill(labels, labels + N, -2);

    std::vector<int> nlabels;
    nlabels.resize(N, -1);

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
        std::fill(distance.begin(), distance.end(), inf);
        for (unsigned ci = 0; ci < centroids.size(); ++ci) {
            const centroid_info& c = centroids[ci];
            const int start_y = int(std::max<float>(0.0, c.y - 2*S));
            const int start_x = int(std::max<float>(0.0, c.x - 2*S));
            const int end_y = int(std::min<float>(Ny, c.y + 2*S));
            const int end_x = int(std::min<float>(Nx, c.x + 2*S));
            assert(start_y < end_y);
            assert(start_x < end_x);
            for (int y = start_y; y != end_y; ++y) {
                for (int x = start_x; x != end_x; ++x) {
                    const int pos = y*Nx + x;
                    float l = array.at(y,x,0);
                    float a = array.at(y,x,1);
                    float b = array.at(y,x,2);
                    const float Ds = diff2(y, c.y) + diff2(x, c.x);
                    const float Dc = diff2(l, c.l) + diff2(a, c.a) + diff2(b, c.b);
                    const float D2 = Dc + Ds*m2S2;

                    assert(D2 < inf);
                    if (D2 < distance[pos]) {
                        distance[pos] = D2;
                        nlabels[pos] = ci;
                    }
                }
            }
        }
        bool changed = false;
        for (int p = 0; p != N; ++p) {
            if (nlabels[p] != labels[p]) {
                labels[p] = nlabels[p];
                changed = true;
            }
        }
        // If nothing changed, we are done
        if (!changed) break;
        std::fill(centroids.begin(), centroids.end(), centroid_info());
        std::fill(centroid_counts.begin(), centroid_counts.end(), 0);
        int y = 0;
        int x = 0;
        for (int pos = 0; pos != N; ++pos) {
            assert(labels[pos] >= 0);
            ++centroid_counts[labels[pos]];
            centroid_info& c = centroids[labels[pos]];
            c.l += array.at(y,x,0);
            c.a += array.at(y,x,1);
            c.b += array.at(y,x,2);
            c.y += y;
            c.x += x;


            ++x;
            if (x == Nx) {
                x = 0;
                ++y;
            }
        }
        for (unsigned ci = 0; ci != centroids.size(); ++ci) {
            centroid_info& c = centroids[ci];
            const int cc =  centroid_counts[ci];
            if (cc) {
                c.l /= cc;
                c.a /= cc;
                c.b /= cc;
                c.y /= cc;
                c.x /= cc;
            }
        }
    }

    for (int i = 0; i != N; ++i) nlabels[i] = i;
    std::vector<int>::iterator nlabelsp = nlabels.begin();

    for (int y = 0; y != Ny; ++y) {
        for (int x = 0; x != Nx; ++x) {
            int nx, ny;
            for (int n = 0; n != 4; ++n) {
                if (fetch_neighbour(n, y, x, ny, nx, Ny, Nx) &&
                    (alabels.at(y,x) == alabels.at(ny,nx))) {
                    const int i =  y*Nx +  x;
                    const int j = ny*Nx + nx;
                    join(nlabelsp, i, j);
                }
            }
        }
    }
    std::vector<bool> is_connected(N);
    for (int y = 0; y != Ny; ++y) {
        for (int x = 0; x != Nx; ++x) {
            const int i = y*Nx + x;
            const int cy = centroids[alabels.at(y,x)].y;
            const int cx = centroids[alabels.at(y,x)].x;
            const int j = cy*Nx + cx;
            is_connected[i] = (find(nlabelsp, i) == find(nlabelsp, j));
        }
    }
    std::vector<bool> seen(N);
    std::priority_queue<SlicPoint> q;
    for (unsigned ci = 0; ci < centroids.size(); ++ci) {
        const centroid_info& c = centroids[ci];
        q.push(SlicPoint(c.y, c.x, ci, 0));
    }
    while (!q.empty()) {
        SlicPoint p = q.top();
        q.pop();
        const int i = p.y*Nx + p.x;
        const centroid_info& c = centroids[p.ci];
        if (seen[i]) continue;
        seen[i] = true;
        if (!is_connected[i]) {
            alabels.at(p.y,p.x) = p.ci;
        }
        for (int n = 0; n != 4; ++n) {
            int ny, nx;
            if (fetch_neighbour(n, p.y, p.x, ny, nx, Ny, Nx)) {
                int j = ny*Nx + nx;
                if (!seen[j] && (!is_connected[j] || alabels.at(ny,nx) == p.ci)) {
                    const int cost = diff2(ny, c.y) + diff2(nx, c.x);
                    q.push(SlicPoint(ny, nx, p.ci, cost));
                }
            }
        }
    }
    // Above, we work with labels in 0..[N-1]. Return should be 1..N
    for (int pos = 0; pos != N; ++pos) ++labels[pos];
    return centroids.size();
}

PyObject* py_slic(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* labels;
    int S;
    float m;
    int max_iters;
    if (!PyArg_ParseTuple(args,"OOifi", &array, &labels, &S, &m, &max_iters)) return NULL;
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
    try {
        if (max_iters < 0) max_iters = 128;
        const int n = slic(numpy::aligned_array<npy_float32>(array), numpy::aligned_array<int>(labels), S, m, max_iters);
        return PyLong_FromLong(n);
    }
    CATCH_PYTHON_EXCEPTIONS
}

PyMethodDef methods[] = {
  {"label",             (PyCFunction)py_label, METH_VARARGS, NULL},
  {"relabel",           (PyCFunction)py_relabel, METH_VARARGS, NULL},
  {"is_same_labeling",  (PyCFunction)py_is_same_labeling, METH_VARARGS, NULL},
  {"remove_regions",    (PyCFunction)py_remove_regions, METH_VARARGS, NULL},
  {"borders",           (PyCFunction)py_borders, METH_VARARGS, NULL},
  {"border",            (PyCFunction)py_border, METH_VARARGS, NULL},
  {"labeled_sum",       (PyCFunction)py_labeled_sum, METH_VARARGS, NULL},
  {"labeled_max_min",   (PyCFunction)py_labeled_max_min, METH_VARARGS, NULL},
  {"slic",              (PyCFunction)py_slic, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
DECLARE_MODULE(_labeled)
