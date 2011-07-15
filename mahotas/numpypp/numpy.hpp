#ifndef MAHOTAS_NUMPYPP_NUMPY_HPP_INCLUDE_GUARD_LPC_
#define MAHOTAS_NUMPYPP_NUMPY_HPP_INCLUDE_GUARD_LPC_
/* Copyright 2010 (C)
 * Luis Pedro Coelho <luis@luispedro.org>
 * License GPL Version 2, or later.
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
npy_intp dtype_code<double>() { return NPY_DOUBLE; }

}

#endif // MAHOTAS_NUMPYPP_NUMPY_HPP_INCLUDE_GUARD_LPC_
