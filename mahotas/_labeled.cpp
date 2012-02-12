#include <map>

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
    for (int i = 0; i != N; ++i, filter.iterate_with(iter), ++iter) {
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

    for (int i = 0; i != N; ++i, fiter.iterate_with(iter), ++iter, ++out) {
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

    for (int ii = 0; ii != N; ++ii, fiter.iterate_with(iter), ++iter, ++out) {
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

template <typename T>
void labeled_sum(numpy::aligned_array<T> array, numpy::aligned_array<int> labeled, T* result, int max) {
    gil_release nogil;
    typename numpy::aligned_array<T>::iterator iterator = array.begin();
    numpy::aligned_array<int>::iterator literator = labeled.begin();
    const int N = array.size();
    for (int i = 0; i != max; ++i) result[i] = 0;
    for (int i = 0; i != N; ++i, ++iterator, ++literator) {
        if ((*literator >= 0) && (*literator < max)) {
            result[*literator] += *iterator;
        }
    }
}


PyObject* py_label(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    if (!PyArg_ParseTuple(args,"OO", &array, &filter)) return NULL;
    if (!PyArray_Check(array) || !PyArray_Check(filter) || PyArray_TYPE(array) != PyArray_TYPE(filter) ||
        !PyArray_ISCARRAY(array) || !PyArray_EquivTypenums(PyArray_TYPE(array), NPY_INT)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    int n = label(numpy::aligned_array<int>(array), numpy::aligned_array<int>(filter));
    PyObject* no = PyInt_FromLong(n);
    Py_INCREF(no);
    return no;
}

PyObject* py_borders(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* filter;
    PyArrayObject* output;
    if (!PyArg_ParseTuple(args,"OOO", &array, &filter, &output)) return NULL;
    if (!PyArray_Check(array) || !PyArray_Check(filter) || PyArray_TYPE(array) != PyArray_TYPE(filter) ||
        !PyArray_Check(output) || PyArray_TYPE(output) != NPY_BOOL || !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    for (int d = 0; d != PyArray_NDIM(array); ++d) {
        if (PyArray_DIM(array, d) != PyArray_DIM(output, d)) {
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
    }
    Py_INCREF(output);


#define HANDLE(type) \
    borders<type>( \
                numpy::aligned_array<type>(array), \
                numpy::aligned_array<type>(filter), \
                numpy::aligned_array<bool>(output));
    SAFE_SWITCH_ON_TYPES_OF(array, false)
#undef HANDLE
    if (PyErr_Occurred()) {
        Py_DECREF(output);
        return NULL;
    }
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
    if (!PyArray_Check(array) || !PyArray_Check(filter) || PyArray_TYPE(array) != PyArray_TYPE(filter) ||
        !PyArray_Check(output) || PyArray_TYPE(output) != NPY_BOOL || !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    for (int d = 0; d != PyArray_NDIM(array); ++d) {
        if (PyArray_DIM(array, d) != PyArray_DIM(output, d)) {
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
    }
    Py_INCREF(output);

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
        Py_DECREF(output);
        return NULL;
    }
    if (always_return || has_any) {
        return PyArray_Return(output);
    }

    Py_DECREF(output);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* py_labeled_sum(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* labeled;
    PyArrayObject* output;
    if (!PyArg_ParseTuple(args,"OOO", &array, &labeled, &output)) return NULL;
    if (!PyArray_Check(array) || !PyArray_Check(labeled) || PyArray_NDIM(array) != PyArray_NDIM(labeled) ||
        !PyArray_EquivTypenums(PyArray_TYPE(labeled), NPY_INT) ||
        !PyArray_Check(output) || PyArray_TYPE(output) != PyArray_TYPE(array) || !PyArray_ISCARRAY(output)) {
        PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
        return NULL;
    }
    for (int d = 0; d != PyArray_NDIM(array); ++d) {
        if (PyArray_DIM(array, d) != PyArray_DIM(labeled, d)) {
            PyErr_SetString(PyExc_RuntimeError, TypeErrorMsg);
            return NULL;
        }
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

    Py_INCREF(Py_None);
    return Py_None;
}

PyMethodDef methods[] = {
  {"label",(PyCFunction)py_label, METH_VARARGS, NULL},
  {"borders",(PyCFunction)py_borders, METH_VARARGS, NULL},
  {"border",(PyCFunction)py_border, METH_VARARGS, NULL},
  {"labeled_sum",(PyCFunction)py_labeled_sum, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_labeled()
  {
    import_array();
    (void)Py_InitModule("_labeled", methods);
  }

