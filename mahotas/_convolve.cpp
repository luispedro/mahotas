// Copyright (C) 2010-2019 Luis Pedro Coelho <luis@luispedro.org>
//
// License: MIT (Check COPYING file)

#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"
#include "_filters.h"

#include <iostream>

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _convolve (which is dangerous: types are not checked!) or a bug in convolve.py.\n";
const char OutputErrorMsg[] =
    "Output type is not valid. "
    "This is caused by either a direct call to _convolve (which is dangerous: types are not checked!) or a bug in convolve.py.\n";


template<typename T>
void convolve1d(const numpy::aligned_array<T> array, const numpy::aligned_array<double> filter, numpy::aligned_array<T> result, ExtendMode mode) {
    gil_release nogil;
    assert(filter.ndims() == 1);
    assert(result.dim(0) == array.dim(0));
    assert(result.dim(1) == array.dim(1));
    assert(array.ndims() == 2);
    assert(result.is_carray());

    const npy_intp N0 = array.dim(0);
    const npy_intp N1 = array.dim(1);
    const npy_intp step = array.stride(1);
    const double* fdata = filter.data();
    const npy_intp Nf = filter.size();
    const npy_intp centre = Nf/2;

    for (npy_intp y = 0; y != N0; ++y) {
        if (centre >= N1) break;
        const T* base0 = array.data(y);
        // The two loops (over x & x_) are almost the same.
        // However, combining them, whilst leading to better code,
        // made the result much slower (probably because there was a need to
        // test the value of x in the inner loop).
        //
        // This was true in a 2013 MacBook Air. Maybe re-check in the future
        T* out = result.data(y,centre);
        for (npy_intp x = centre; x != (N1 - centre); ++x) {
            double cur = 0;
            for (npy_intp j = 0; j != Nf; ++j) {
                const double val = base0[(x + j-centre)*step];
                assert(val == array.at(y, x - centre + j));
                cur += val * fdata[j];
            }
            *out++ = T(cur); // consider T(std::round(cur)) as well
        }
    }

    std::vector<npy_intp> offsets;
    offsets.resize(Nf);
    for (npy_intp x_ = 0; x_ != 2*centre && x_ < N1; ++x_) {
        const npy_intp x = (x_ < centre ? x_ : (N1 - 1) - (x_ - centre));
        for (npy_intp j = 0; j != Nf; ++j) {
            offsets[j] = fix_offset(mode, x + (j - centre), N1);
        }
        for (npy_intp y = 0; y != N0; ++y) {
            const T* base0 = array.data(y);
            double cur = 0;
            for (npy_intp j = 0; j != Nf; ++j) {
                const double val = (offsets[j] == border_flag_value ? 0 : base0[offsets[j] * step]);
                cur += val * fdata[j];
            }
            *result.data(y,x) = T(cur); // as above T(std::round(cur)) could be better
        }
    }
}


template<typename T>
void convolve(const numpy::aligned_array<T> array, const numpy::aligned_array<T> filter, numpy::aligned_array<T> result, int mode) {
    gil_release nogil;
    const npy_intp N = array.size();
    typename numpy::aligned_array<T>::const_iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), filter.raw_array(), ExtendMode(mode), true);
    const npy_intp N2 = fiter.size();
    T* out = result.data();

    for (npy_intp i = 0; i != N; ++i, fiter.iterate_both(iter), ++out) {
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
        for (npy_intp j = 0; j != N2; ++j) {
            T val;
            if (fiter.retrieve(iter, j, val)) {
                cur += double(val)*fiter[j];
            }
        }
        *out = T(cur); // again, possibly T(std::round(cur))
    }
}


PyObject* py_convolve1d(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    PyArrayObject* output;
    int mode;
    if (!PyArg_ParseTuple(args,"OOOi", &array, &filter, &output, &mode)) return NULL;
    if (!numpy::are_arrays(array, filter, output) ||
            !numpy::same_shape(output, array) ||
            !numpy::equiv_typenums(output, array) ||
            !PyArray_ISCARRAY(output)) {
            PyErr_SetString(PyExc_RuntimeError, OutputErrorMsg);
            return NULL;
    }
    holdref outref(output);

#define HANDLE(type) \
    convolve1d<type>(numpy::aligned_array<type>(array), numpy::aligned_array<double>(filter), numpy::aligned_array<type>(output), ExtendMode(mode));

    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(output);
    return PyArray_Return(output);
}

PyObject* py_convolve(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    PyArrayObject* output;
    int mode;
    if (!PyArg_ParseTuple(args,"OOOi", &array, &filter, &output, &mode)) return NULL;
    if (!numpy::are_arrays(array, filter) ||
        !numpy::equiv_typenums(array, filter) ||
        PyArray_NDIM(array) != PyArray_NDIM(filter)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }

    if (reinterpret_cast<PyObject*>(output) == Py_None) {
        output = reinterpret_cast<PyArrayObject*>(
                PyArray_EMPTY(PyArray_NDIM(array), PyArray_DIMS(array), PyArray_TYPE(array), 0));
        if (!output) return NULL;
    } else {
        if (!PyArray_Check(output) ||
            !numpy::same_shape(output, array) ||
            !numpy::equiv_typenums(output, array) ||
            !PyArray_ISCARRAY(output)) {
            PyErr_SetString(PyExc_RuntimeError, OutputErrorMsg);
            return NULL;
        }
        Py_INCREF(output);
    }

#define HANDLE(type) \
    convolve<type>(numpy::aligned_array<type>(array), numpy::aligned_array<type>(filter), numpy::aligned_array<type>(output), mode);
    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE
    return PyArray_Return(output);
}

template <typename T>
void haar(numpy::aligned_array<T> array) {
    gil_release nogil;
    const npy_intp N0 = array.dim(0);
    const npy_intp N1 = array.dim(1);
    const npy_intp step = array.stride(1);

    std::vector<T> bufdata;
    bufdata.resize(N1);
    T* buffer = &bufdata[0];
    T* low = buffer;
    T* high = buffer + N1/2;

    for (npy_intp y = 0; y != N0; ++y) {
        T* data = array.data(y);
        for (npy_intp x = 0; x != (N1/2); ++x) {
            const T di = data[2*x*step];
            const T di1 = data[(2*x + 1)*step];
            low[x] = di + di1;
            high[x] = di1 - di;
        }
        for (npy_intp x = 0; x != N1; ++x) {
            data[step*x] = buffer[x];
        }
    }
}

template<typename T>
T _access(const T* data, const npy_intp N, const npy_intp p, const npy_intp step) {
    if (p < 0) return T();
    if (p >= N) return T();
    return data[p*step];
}

template <typename T>
void wavelet(numpy::aligned_array<T> array, const float coeffs[], const int ncoeffs) {
    gil_release nogil;
    const npy_intp N0 = array.dim(0);
    const npy_intp N1 = array.dim(1);
    const npy_intp step = array.stride(1);

    std::vector<T> bufdata;
    bufdata.resize(N1);
    T* buffer = &bufdata[0];
    T* low = buffer;
    T* high = buffer + N1/2;

    for (npy_intp y = 0; y != N0; ++y) {
        T* data = array.data(y);
        for (npy_intp x = 0; x < (N1/2); ++x) {
            T l = T();
            T h = T();
            bool even = true;
            for (npy_intp ci = 0; ci != ncoeffs; ++ci) {
                T val = _access(data, N1, 2*x+ci, step);
                const float cl = coeffs[ncoeffs-ci-1];
                const float ch = (even ? -1 : +1) * coeffs[ci];
                l += cl*val;
                h += ch*val;
                even = !even;
            }

            low[x] = l;
            high[x] = h;
        }

        for (npy_intp x = 0; x != N1; ++x) {
            data[step*x] = buffer[x];
        }
    }
}

inline
bool _is_even(npy_intp x) { return (x & 1) == 0; }

template <typename T>
void iwavelet(numpy::aligned_array<T> array, const float coeffs[], const int ncoeffs) {
    gil_release nogil;
    const npy_intp N0 = array.dim(0);
    const npy_intp N1 = array.dim(1);
    const npy_intp step = array.stride(1);

    std::vector<T> bufdata;
    bufdata.resize(N1);
    T* buffer = &bufdata[0];
    for (npy_intp y = 0; y != N0; ++y) {
        T* data = array.data(y);
        T* low = data;
        T* high = data + step*N1/2;
        for (npy_intp x = 0; x < N1; ++x) {
            T l = T();
            T h = T();
            for (npy_intp ci = 0; ci != ncoeffs; ++ci) {
                const int xmap2 = x+ci-ncoeffs+2;
                if (!_is_even(xmap2)) {
                    const int xmap = xmap2 / 2;
                    const float cl = coeffs[ci];
                    const float ch = (_is_even(ci) ? +1 : -1) * coeffs[ncoeffs-ci-1];
                    l += cl*_access( low, N1/2, xmap, step);
                    h += ch*_access(high, N1/2, xmap, step);
                }
            }
            buffer[x] = (l+h)/2.;
        }

        for (int x = 0; x != N1; ++x) {
            data[step*x] = buffer[x];
        }
    }
}

PyObject* py_haar(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "O", &array) ||
        !PyArray_Check(array) || PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }

#define HANDLE(type) \
        haar<type>(numpy::aligned_array<type>(array));

    SAFE_SWITCH_ON_FLOAT_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(array);
    return PyArray_Return(array);
}

// These are the Daubechie coefficients
// This is the scaling function, the wavelet is multiplication with (-1)^k
// These values were copy&pasted from wikipedia
const float D2[] = { 1.,  1. };
const float D4[] = { 0.6830127,  1.1830127,  0.3169873, -0.1830127 };
const float D6[] = { 0.47046721,  1.14111692,  0.650365  , -0.19093442, -0.12083221,
                        0.0498175  };
const float D8[] = { 0.32580343,  1.01094572,  0.8922014 , -0.03957503, -0.26450717,
                    0.0436163 ,  0.0465036 , -0.01498699 };
const float D10[] = { 0.22641898,  0.85394354,  1.02432694,  0.19576696, -0.34265671,
                    -0.04560113,  0.10970265, -0.0088268 , -0.01779187,  0.00471743 };
const float D12[] = {  1.57742430e-01,   6.99503810e-01,   1.06226376e+00,
                     4.45831320e-01,  -3.19986600e-01,  -1.83518060e-01,
                     1.37888090e-01,   3.89232100e-02,  -4.46637500e-02,
                     7.83251152e-04,   6.75606236e-03,  -1.52353381e-03 };
const float D14[] = {  1.10099430e-01,   5.60791280e-01,   1.03114849e+00,
                     6.64372480e-01,  -2.03513820e-01,  -3.16835010e-01,
                     1.00846700e-01,   1.14003450e-01,  -5.37824500e-02,
                    -2.34399400e-02,   1.77497900e-02,   6.07514995e-04,
                    -2.54790472e-03,   5.00226853e-04 };
const float D16[] = {  7.69556200e-02,   4.42467250e-01,   9.55486150e-01,
                     8.27816530e-01,  -2.23857400e-02,  -4.01658630e-01,
                     6.68194092e-04,   1.82076360e-01,  -2.45639000e-02,
                    -6.23502100e-02,   1.97721600e-02,   1.23688400e-02,
                    -6.88771926e-03,  -5.54004549e-04,   9.55229711e-04,
                    -1.66137261e-04 };
const float D18[] = {  5.38503500e-02,   3.44834300e-01,   8.55349060e-01,
                     9.29545710e-01,   1.88369550e-01,  -4.14751760e-01,
                    -1.36953550e-01,   2.10068340e-01,   4.34526750e-02,
                    -9.56472600e-02,   3.54892813e-04,   3.16241700e-02,
                    -6.67962023e-03,  -6.05496058e-03,   2.61296728e-03,
                     3.25814671e-04,  -3.56329759e-04,   5.56455140e-05 };
const float D20[] = {  3.77171600e-02,   2.66122180e-01,   7.45575070e-01,
                     9.73628110e-01,   3.97637740e-01,  -3.53336200e-01,
                    -2.77109880e-01,   1.80127450e-01,   1.31602990e-01,
                    -1.00966570e-01,  -4.16592500e-02,   4.69698100e-02,
                     5.10043697e-03,  -1.51790000e-02,   1.97332536e-03,
                     2.81768659e-03,  -9.69947840e-04,  -1.64709006e-04,
                     1.32354367e-04,  -1.87584100e-05 };


PyObject* py_wavelet(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* coeffs;
    if (!PyArg_ParseTuple(args, "OO", &array, &coeffs) ||
        !numpy::are_arrays(array, coeffs) ||
        PyArray_NDIM(array) != 2 ||
        PyArray_TYPE(coeffs) != NPY_FLOAT ||
        !PyArray_ISCARRAY(coeffs)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    numpy::aligned_array<float> acoeffs(coeffs);

#define HANDLE(type) \
        wavelet<type>(numpy::aligned_array<type>(array), acoeffs.data(), acoeffs.dim(0));

    SAFE_SWITCH_ON_FLOAT_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(array);
    return PyArray_Return(array);
}

PyObject* py_iwavelet(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* coeffs;
    if (!PyArg_ParseTuple(args, "OO", &array, &coeffs) ||
        !numpy::are_arrays(array, coeffs) ||
        PyArray_NDIM(array) != 2 ||
        PyArray_TYPE(coeffs) != NPY_FLOAT ||
        !PyArray_ISCARRAY(coeffs)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    numpy::aligned_array<float> acoeffs(coeffs);

#define HANDLE(type) \
        iwavelet<type>(numpy::aligned_array<type>(array), acoeffs.data(), acoeffs.dim(0));

    SAFE_SWITCH_ON_FLOAT_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(array);
    return PyArray_Return(array);
}

const float* dcoeffs(const int code) {
    switch (code) {
        case 0: return D2;
        case 1: return D4;
        case 2: return D6;
        case 3: return D8;
        case 4: return D10;
        case 5: return D12;
        case 6: return D14;
        case 7: return D16;
        case 8: return D18;
        case 9: return D20;
        default:
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
}
PyObject* py_daubechies(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    int code;
    if (!PyArg_ParseTuple(args, "Oi", &array, &code) ||
        !numpy::are_arrays(array) ||
        PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    const float* coeffs = dcoeffs(code);
    int ncoeffs = 2*(code + 1);
    if (!coeffs) return NULL;

#define HANDLE(type) \
        wavelet<type>(numpy::aligned_array<type>(array), coeffs, ncoeffs);

    SAFE_SWITCH_ON_FLOAT_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(array);
    return PyArray_Return(array);
}

PyObject* py_idaubechies(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    int code;
    if (!PyArg_ParseTuple(args, "Oi", &array, &code) ||
        !numpy::are_arrays(array) ||
        PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    const float* coeffs = dcoeffs(code);
    int ncoeffs = 2*(code + 1);
    if (!coeffs) return NULL;

    Py_INCREF(array);
#define HANDLE(type) \
        iwavelet<type>(numpy::aligned_array<type>(array), coeffs, ncoeffs);

    SAFE_SWITCH_ON_FLOAT_TYPES_OF(array);
#undef HANDLE

    return PyArray_Return(array);
}

template <typename T>
void ihaar(numpy::aligned_array<T> array) {
    gil_release nogil;
    const int N0 = array.dim(0);
    const int N1 = array.dim(1);
    const int step = array.stride(1);

    std::vector<T> bufdata;
    bufdata.resize(N1);
    T* buffer = &bufdata[0];

    for (int y = 0; y != N0; ++y) {
        T* data = array.data(y);
        T* low = data;
        T* high = data + step*N1/2;
        for (int x = 0; x != (N1/2); ++x) {
            const T h = high[x*step];
            const T l = low[x*step];
            buffer[2*x]   = (l-h)/2;
            buffer[2*x+1] = (l+h)/2;
        }
        for (int x = 0; x != N1; ++x) {
            data[step*x] = buffer[x];
        }
    }
}


PyObject* py_ihaar(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "O", &array) ||
        !PyArray_Check(array) || PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }

#define HANDLE(type) \
        ihaar<type>(numpy::aligned_array<type>(array));

    SAFE_SWITCH_ON_FLOAT_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(array);
    return PyArray_Return(array);
}

template<typename T>
void rank_filter(numpy::aligned_array<T> res, const numpy::aligned_array<T> array, const numpy::aligned_array<T> Bc, const int rank, const int mode, const T cval = T()) {
    gil_release nogil;
    const npy_intp N = res.size();
    typename numpy::aligned_array<T>::const_iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), Bc.raw_array(), ExtendMode(mode), true);
    const npy_intp N2 = fiter.size();
    if (rank < 0 || rank >= N2) {
        return;
    }
    std::vector<T> n_data;
    n_data.resize(N2);
    // T* is a fine iterator type.
    T* rpos = res.data();

    // This is generally a T*, except in debug builds, so we get checking there
    typename std::vector<T>::iterator neighbours = n_data.begin();

    for (npy_intp i = 0; i != N; ++i, ++rpos, fiter.iterate_both(iter)) {
        npy_intp n = 0;
        for (npy_intp j = 0; j != N2; ++j) {
            T val;
            if (fiter.retrieve(iter, j, val)) neighbours[n++] = val;
            else if (mode == ExtendConstant) neighbours[n++] = cval;
        }
        npy_intp currank = rank;
        if (n != N2) {
            currank = npy_intp(n * rank/double(N2));
        }
        std::nth_element(neighbours, neighbours + currank, neighbours + n);
        *rpos = neighbours[currank];
    }
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
        PyArray_NDIM(array) != PyArray_NDIM(Bc) ||
        !PyArray_EquivTypenums(PyArray_TYPE(array), PyArray_TYPE(output)) ||
        !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    holdref r(output);

#define HANDLE(type) \
        rank_filter<type>(numpy::aligned_array<type>(output), numpy::aligned_array<type>(array), numpy::aligned_array<type>(Bc), rank, mode);
    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(output);
    return PyArray_Return(output);
}

template<typename T>
void mean_filter(numpy::aligned_array<double> res, const numpy::aligned_array<T> array, const numpy::aligned_array<T> Bc, const int mode, const double cval) {
    gil_release nogil;
    const int N = res.size();
    typename numpy::aligned_array<T>::const_iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), Bc.raw_array(), ExtendMode(mode), true);
    const int N2 = fiter.size();
    double* rpos = res.data();

    for (int i = 0; i != N; ++i, ++rpos, fiter.iterate_both(iter)) {
        int n = N2;
        double sum = 0;
        for (int j = 0; j != N2; ++j) {
            T val;
            if (fiter.retrieve(iter, j, val)) sum += val;
            else if (mode == ExtendConstant) sum += cval;
            else --n;
        }
        *rpos = sum/n;
    }
}
PyObject* py_mean_filter(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* Bc;
    PyArrayObject* output;
    int mode;
    double cval;
    if (!PyArg_ParseTuple(args, "OOOid", &array, &Bc, &output, &mode, &cval) ||
        !numpy::are_arrays(array, Bc, output) ||
        !numpy::equiv_typenums(array, Bc) ||
        !numpy::check_type<double>(output) ||
        !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    holdref r(output);

#define HANDLE(type) \
        mean_filter<type>(numpy::aligned_array<double>(output), numpy::aligned_array<type>(array), numpy::aligned_array<type>(Bc), mode, cval);
    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(output);
    return PyArray_Return(output);
}

template <typename T>
void template_match(numpy::aligned_array<T> res, const numpy::aligned_array<T> f, const numpy::aligned_array<T> t, int mode, bool just_equality) {
    gil_release nogil;
    const npy_intp N = res.size();
    typename numpy::aligned_array<T>::const_iterator iter = f.begin();
    filter_iterator<T> fiter(f.raw_array(), t.raw_array(), ExtendMode(mode), false);
    const npy_intp N2 = fiter.size();
    assert(res.is_carray());
    // T* is a fine iterator type.
    T* rpos = res.data();

    for (npy_intp i = 0; i != N; ++i, ++rpos, fiter.iterate_both(iter)) {
        T diff2 = T(0);
        for (npy_intp j = 0; j != N2; ++j) {
            T val;
            if (fiter.retrieve(iter, j, val)) {
                const T tj = fiter[j];
                const T delta = (val > tj ? val - tj : tj - val);
                if (just_equality && delta) {
                    diff2 = 1;
                    break;
                }
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
    int just_equality;
    PyArrayObject* output;
    if (!PyArg_ParseTuple(args, "OOOii", &array, &template_, &output, &mode, &just_equality)) return NULL;
    if (!numpy::are_arrays(array, template_, output) ||
        !numpy::equiv_typenums(array, template_, output) ||
        !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    holdref r(output);

#define HANDLE(type) \
        template_match<type>(numpy::aligned_array<type>(output), numpy::aligned_array<type>(array), numpy::aligned_array<type>(template_), mode, just_equality);
    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(output);
    return PyArray_Return(output);
}

template <typename T>
void find2d(const numpy::aligned_array<T> array, const numpy::aligned_array<T> target, numpy::aligned_array<bool> out) {
    gil_release nogil;
    const npy_intp N0 = array.dim(0);
    const npy_intp N1 = array.dim(1);

    const npy_intp Nt0 = target.dim(0);
    const npy_intp Nt1 = target.dim(1);
    assert(out.is_carray());
    bool* rpos = out.data();
    std::fill(rpos, rpos + N0*N1, false);

    for (npy_intp y = 0; y < N0 - Nt0; ++y) {
        for (npy_intp x = 0; x < N1 - Nt1; ++x) {
            for (npy_intp sy = 0; sy < Nt0; ++sy) {
                for (npy_intp sx = 0; sx < Nt1; ++sx) {
                    if (array.at(y + sy,x + sx) != target.at(sy,sx)) {
                        goto next_pos;
                    }
                }
            }
            out.at(y, x) = true;
            next_pos:
                ;
        }
    }
}

PyObject* py_find2d(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* target;
    PyArrayObject* output;
    if (!PyArg_ParseTuple(args,"OOO", &array, &target, &output)) return NULL;
    if (!numpy::are_arrays(array, target, output) ||
            !numpy::same_shape(output, array) ||
            !numpy::equiv_typenums(array, target) ||
            !numpy::check_type<bool>(output) ||
            !PyArray_ISCARRAY(output)) {
            PyErr_SetString(PyExc_RuntimeError, OutputErrorMsg);
            return NULL;
    }
    holdref outref(output);

#define HANDLE(type) \
    find2d<type>(numpy::aligned_array<type>(array), numpy::aligned_array<type>(target), numpy::aligned_array<bool>(output));

    SAFE_SWITCH_ON_TYPES_OF(array);
#undef HANDLE

    Py_INCREF(output);
    return PyArray_Return(output);
}


PyMethodDef methods[] = {
  {"convolve",(PyCFunction)py_convolve, METH_VARARGS, NULL},
  {"convolve1d",(PyCFunction)py_convolve1d, METH_VARARGS, NULL},
  {"wavelet",(PyCFunction)py_wavelet, METH_VARARGS, NULL},
  {"iwavelet",(PyCFunction)py_iwavelet, METH_VARARGS, NULL},
  {"daubechies",(PyCFunction)py_daubechies, METH_VARARGS, NULL},
  {"idaubechies",(PyCFunction)py_idaubechies, METH_VARARGS, NULL},
  {"haar",(PyCFunction)py_haar, METH_VARARGS, NULL},
  {"ihaar",(PyCFunction)py_ihaar, METH_VARARGS, NULL},
  {"rank_filter",(PyCFunction)py_rank_filter, METH_VARARGS, NULL},
  {"mean_filter",(PyCFunction)py_mean_filter, METH_VARARGS, NULL},
  {"template_match",(PyCFunction)py_template_match, METH_VARARGS, NULL},
  {"find2d",(PyCFunction)py_find2d, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace

DECLARE_MODULE(_convolve)

