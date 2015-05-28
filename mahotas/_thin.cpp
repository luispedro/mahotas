#include <cstring>
#include <iostream>
#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"
#include "utils.hpp"

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _thin (which is dangerous: types are not checked!) or a bug in mahotas.\n";


const int Element_Size = 6;
struct structure_element {
    bool data[Element_Size];
    npy_intp offset[Element_Size];
};

inline
bool match(const bool* array, const structure_element& elem) {
    if (!*array) return false;
    for (int i = 0; i != Element_Size; ++i) {
        if (elem.data[i] != *(array+elem.offset[i])) return false;
    }
    return true;
}

inline
void fast_hitmiss(PyArrayObject* array, const structure_element& elem, PyArrayObject* outputa) {
    const bool* first = reinterpret_cast<const bool*>(PyArray_DATA(array));
    const bool* const last = first + PyArray_NBYTES(array)/sizeof(bool);
    bool* output = reinterpret_cast<bool*>(PyArray_DATA(outputa));
    for ( ; first != last; ++first) {
        *output++ = match(first, elem);
    }
}

inline
npy_intp coordinates_delta(PyArrayObject* array, npy_intp d0, npy_intp d1) {
    return (d0*PyArray_STRIDE(array,0) + d1*PyArray_STRIDE(array,1))/sizeof(bool);
}

const bool boolvals[] =    { false, false, false, true, true, true };
// edge
const npy_intp edelta0[] = {    -1,    -1,    -1,   +1,   +1,   +1 };
const npy_intp edelta1[] = {    -1,     0,    +1,   -1,    0,   +1 };
// corner 1
const npy_intp cdelta0[] = {    -1,    -1,     0,    0,   +1,   +1 };
const npy_intp cdelta1[] = {    -1,     0,    -1,   +1,    0,   +1 };
// corner 2
const npy_intp adelta0[] = {    -1,    -1,     0,    0,   +1,   +1 };
const npy_intp adelta1[] = {     0,    +1,    +1,   -1,   -1,    0 };


// This is useful for debugging purposes, but otherwise unused:
void show_data(const bool flip, const npy_intp* delta0, const npy_intp* delta1) {
    int arr[3][3];
    for (int j = 0; j!= 3; ++j)
        for (int k = 0; k != 3; ++k)
            arr[j][k] = 2;
    for (int j = 0; j != Element_Size; ++j) {
        arr[delta0[j] + 1][delta1[j] + 1] = (flip ? ! boolvals[j] : boolvals[j]);
    }
    arr[1][1] = 1;
    for (int j = 0; j!= 3; ++j) {
        for (int k = 0; k != 3; ++k) std::cout << arr[j][k];
        std:: cout << '\n';
    }
    std:: cout << '\n';
}

void fill_data(PyArrayObject* array, structure_element& elem, const bool flip, const npy_intp* delta0, const npy_intp* delta1) {
    //show_data(flip, delta0, delta1);
    for (int j = 0; j != Element_Size; ++j) {
        elem.data[j] = (flip ? ! boolvals[j]: boolvals[j]);
        elem.offset[j] = coordinates_delta(array, delta0[j], delta1[j]);
    }
}


PyObject* py_thin(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* buffer;
    int max_iter;
    if (!PyArg_ParseTuple(args,"OOi", &array, &buffer, &max_iter)) return NULL;
    if (!numpy::are_arrays(array, buffer) ||
        !numpy::check_type<bool>(array) ||
        !numpy::check_type<bool>(buffer) ||
        !numpy::same_shape(array, buffer) ||
        !PyArray_ISCONTIGUOUS(array) || !PyArray_ISCONTIGUOUS(buffer)) {
            PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
            return NULL;
    }
    { // DROP THE GIL
        gil_release nogil;
        const int Nr_Elements = 8;
        structure_element elems[Nr_Elements];
        fill_data(array, elems[0], false, edelta0, edelta1);
        fill_data(array, elems[1], false, adelta0, adelta1);
        fill_data(array, elems[2],  true, edelta1, edelta0);
        fill_data(array, elems[3],  true, cdelta0, cdelta1);
        fill_data(array, elems[4],  true, edelta0, edelta1);
        fill_data(array, elems[5],  true, adelta0, adelta1);
        fill_data(array, elems[6], false, cdelta0, cdelta1);
        fill_data(array, elems[7], false, edelta1, edelta0);


        const npy_int N = PyArray_SIZE(array);
        bool any_change = true;
        int n = 0;
        while (any_change && ((max_iter < 0) || n++ < max_iter)) {
            any_change = false;
            for (int i = 0; i != Nr_Elements; ++i) {
                fast_hitmiss(array, elems[i], buffer);
                bool* pa = reinterpret_cast<bool*>(PyArray_DATA(array));
                const bool* pb = reinterpret_cast<bool*>(PyArray_DATA(buffer));
                for (int j = 0; j != N; ++j) {
                    if (*pb && *pa) {
                        *pa = false;
                        any_change = true;
                    }
                    ++pa;
                    ++pb;
                }
            }
        }

    }

    Py_INCREF(array);
    return PyArray_Return(array);
}


PyMethodDef methods[] = {
  {"thin",(PyCFunction)py_thin, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
DECLARE_MODULE(_thin)
