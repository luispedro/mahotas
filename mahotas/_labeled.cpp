// Copyright (C) 2010-2012  Luis Pedro Coelho <luis@luispedro.org>
//
// License: MIT (see COPYING file)
#include <map>
#include <functional>

#include "numpypp/array.hpp"
#include "numpypp/numpy.hpp"
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
    "This is caused by either a direct call to _labeled (which is dangerous: types are not checked!) or a bug in labeled.py.\n";


// This is a standard union-find structure
int find(int* data, int i) {
    if (data[i] == i) return i;
    int j = find(data, data[i]);
    data[i] = j;
    return j;
}
void compress(int* data, int i) {
    find(data,i);
}


void join(int* data, int i, int j) {
    i = find(data, i);
    j = find(data, j);
    assert(i >= 0);
    assert(j >= 0);
    data[i] = j;
}

int label(numpy::aligned_array<int> labeled, numpy::aligned_array<int> Bc) {
    gil_release nogil;
    const int N = labeled.size();
    int* data = labeled.data();
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




template<typename T>
void borders(numpy::aligned_array<T> array, numpy::aligned_array<T> filter, numpy::aligned_array<bool> result) {
    gil_release nogil;
    const int N = array.size();
    typename numpy::aligned_array<T>::iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), filter.raw_array(), EXTEND_CONSTANT, true);
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
bool border(numpy::aligned_array<T> array, numpy::aligned_array<T> filter, numpy::aligned_array<bool> result, T i, T j) {
    gil_release nogil;
    const int N = array.size();
    typename numpy::aligned_array<T>::iterator iter = array.begin();
    filter_iterator<T> fiter(array.raw_array(), filter.raw_array(), EXTEND_CONSTANT, true);
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
void labeled_foldl(numpy::aligned_array<T> array, numpy::aligned_array<int> labeled, T* result, const int maxlabel, const T start, F f) {
    gil_release nogil;
    typename numpy::aligned_array<T>::iterator iterator = array.begin();
    numpy::aligned_array<int>::iterator literator = labeled.begin();
    const int N = array.size();
    std::fill(result, result + maxlabel, start);
    for (int i = 0; i != N; ++i, ++iterator, ++literator) {
        if ((*literator >= 0) && (*literator < maxlabel)) {
            result[*literator] = f(*iterator, result[*literator]);
        }
    }
}

template <typename T>
void labeled_sum(numpy::aligned_array<T> array, numpy::aligned_array<int> labeled, T* result, const int maxlabel) {
    labeled_foldl(array, labeled, result, maxlabel, T(), std::plus<T>());
}
template <>
void labeled_sum<bool>(numpy::aligned_array<bool> array, numpy::aligned_array<int> labeled, bool* result, const int maxlabel) {
    labeled_foldl(array, labeled, result, maxlabel, false, std::logical_or<bool>());
}

template <typename T>
void labeled_max(numpy::aligned_array<T> array, numpy::aligned_array<int> labeled, T* result, const int maxlabel) {
    labeled_foldl(array, labeled, result, maxlabel,std::numeric_limits<T>::min(), std::max<T>);
}

template <typename T>
void labeled_min(numpy::aligned_array<T> array, numpy::aligned_array<int> labeled, T* result, const int maxlabel) {
    labeled_foldl(array, labeled, result, maxlabel,std::numeric_limits<T>::max(), std::min<T>);
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

PyObject* py_borders(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    PyArrayObject* output;
    if (!PyArg_ParseTuple(args,"OOO", &array, &filter, &output)) return NULL;
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
                numpy::aligned_array<bool>(output));
    SAFE_SWITCH_ON_TYPES_OF(array, false)
#undef HANDLE
    if (PyErr_Occurred()) {
        return NULL;
    }
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
    has_any = border<type>( \
                numpy::aligned_array<type>(array), \
                numpy::aligned_array<type>(filter), \
                numpy::aligned_array<bool>(output), \
                static_cast<type>(i), \
                static_cast<type>(j));
    SAFE_SWITCH_ON_TYPES_OF(array, false);
#undef HANDLE
    if (PyErr_Occurred()) {
        return NULL;
    }
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
        type* odata = static_cast<type*>(PyArray_DATA(output)); \
        labeled_sum<type>( \
                numpy::aligned_array<type>(array), \
                numpy::aligned_array<int>(labeled), \
                odata, \
                maxi); \
    }
    SAFE_SWITCH_ON_TYPES_OF(array, true);
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
        type* odata = static_cast<type*>(PyArray_DATA(output)); \
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
    SAFE_SWITCH_ON_TYPES_OF(array, true);
#undef HANDLE

    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
  {"label",(PyCFunction)py_label, METH_VARARGS, NULL},
  {"borders",(PyCFunction)py_borders, METH_VARARGS, NULL},
  {"border",(PyCFunction)py_border, METH_VARARGS, NULL},
  {"labeled_sum",(PyCFunction)py_labeled_sum, METH_VARARGS, NULL},
  {"labeled_max_min",(PyCFunction)py_labeled_max_min, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
DECLARE_MODULE(_labeled)
