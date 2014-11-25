#ifndef MAHOTAS_NUMPYPP_NUMPY_HPP_INCLUDE_GUARD_LPC_
#define MAHOTAS_NUMPYPP_NUMPY_HPP_INCLUDE_GUARD_LPC_
/* Copyright 2010-2014 (C)
 * Luis Pedro Coelho <luis@luispedro.org>
 * License: MIT
 */

#include <complex>

#include <Python.h>
#include <numpy/ndarrayobject.h>

namespace numpy {

template <typename T>
inline
npy_intp dtype_code();

#define DECLARE_DTYPE_CODE(type, constant) \
    template <> inline \
    npy_intp dtype_code<type>() { return constant; } \
    \
    template <> inline \
    npy_intp dtype_code<const type>() { return constant; } \
    \
    template <> inline \
    npy_intp dtype_code<volatile type>() { return constant; } \
    \
    template <> inline \
    npy_intp dtype_code<volatile const type>() { return constant; }

DECLARE_DTYPE_CODE(bool, NPY_BOOL)
DECLARE_DTYPE_CODE(float, NPY_FLOAT)
DECLARE_DTYPE_CODE(char, NPY_BYTE)
DECLARE_DTYPE_CODE(unsigned char, NPY_UBYTE)
DECLARE_DTYPE_CODE(short, NPY_SHORT)
DECLARE_DTYPE_CODE(unsigned short, NPY_USHORT)
DECLARE_DTYPE_CODE(int, NPY_INT)
DECLARE_DTYPE_CODE(long, NPY_LONG)
DECLARE_DTYPE_CODE(unsigned long, NPY_ULONG)
DECLARE_DTYPE_CODE(long long, NPY_LONGLONG)
DECLARE_DTYPE_CODE(unsigned long long, NPY_ULONGLONG)
DECLARE_DTYPE_CODE(double, NPY_DOUBLE)
#if defined(NPY_FLOAT128)
DECLARE_DTYPE_CODE(npy_float128, NPY_FLOAT128)
#endif /* NPY_FLOAT128 */
DECLARE_DTYPE_CODE(std::complex<float>, NPY_CFLOAT)
DECLARE_DTYPE_CODE(std::complex<double>, NPY_CDOUBLE)
DECLARE_DTYPE_CODE(unsigned int, NPY_UINT)

template<typename T>
bool check_type(PyArrayObject* a) { return !!PyArray_EquivTypenums(PyArray_TYPE(a), dtype_code<T>()); }

template<typename T>
bool check_type(PyObject* a) { return check_type<T>(reinterpret_cast<PyArrayObject*>(a)); }


template<typename T>
struct no_ptr { typedef T type; };
template<typename T>
struct no_ptr<T*> { typedef T type; };
template<typename T>
struct no_ptr<const T*> { typedef T type; };

template<typename T>
T ndarray_cast(PyArrayObject* a) {
    assert(check_type<typename no_ptr<T>::type>(a));
    assert(PyArray_ISALIGNED(a));
    // The reason for the intermediate ``as_voidp`` variable is the following:
    // around version 1.7.0, numpy played with having ``PyArray_DATA`` return
    // ``char*`` as opposed to ``void*``. ``reinterpret_cast`` from void* was
    // actually against the standard in C++ pre C++11 (although G++ accepts
    // it).
    //
    // This roundabout way works on all these versions.
    void* as_voidp = PyArray_DATA(a);
    return const_cast<T>(static_cast<T>(as_voidp));
}
template<typename T>
T ndarray_cast(PyObject* pa) {
    assert(PyArray_Check(pa));
    return ndarray_cast<T>((PyArrayObject*)pa);
}

}

#endif // MAHOTAS_NUMPYPP_NUMPY_HPP_INCLUDE_GUARD_LPC_
