#ifndef MAHOTAS_NUMPYPP_NUMPY_HPP_INCLUDE_GUARD_LPC_
#define MAHOTAS_NUMPYPP_NUMPY_HPP_INCLUDE_GUARD_LPC_
/* Copyright 2010 (C)
 * Luis Pedro Coelho <luis@luispedro.org>
 * License: MIT
 */


extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace numpy {

template <typename T>
inline
npy_intp dtype_code();

template <>
inline
npy_intp dtype_code<bool>() { return NPY_BOOL; }

template <>
inline
npy_intp dtype_code<float>() { return NPY_FLOAT; }

template <>
inline
npy_intp dtype_code<int>() { return NPY_INT; }

template <>
inline
npy_intp dtype_code<long>() { return NPY_LONG; }

template <>
inline
npy_intp dtype_code<double>() { return NPY_DOUBLE; }

template<typename T>
bool check_type(PyArrayObject* a) { return PyArray_EquivTypenums(PyArray_TYPE(a), dtype_code<T>()); }

template<typename T>
bool check_type(PyObject* a) { return check_type<T>(reinterpret_cast<PyArrayObject*>(a)); }

}

#endif // MAHOTAS_NUMPYPP_NUMPY_HPP_INCLUDE_GUARD_LPC_
