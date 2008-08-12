#include "array.hpp"
#include "dispatch.hpp"
using namespace numpy_utils;

template<typename T>
void dilate(numpy_aligned_array<T> res, numpy_array<T> array, numpy_aligned_array<T> Bc) {
    const unsigned N = res.size();
    const unsigned N2 = Bc.size();

    typename numpy_array<T>::iterator pos = array.begin();
    for (int i = 0; i != N; ++i, ++pos) {
        typename numpy_aligned_array<T>::iterator startc = Bc.begin();
        bool on = true;
        for (int j = 0; j != N2; ++j, ++startc) {
            if (*startc) {
                numpy_position npos = pos.position() + startc.position();
                if (array.validposition(npos) && !array.at(npos)) {
                    on = false;
                    break;
                }
            }
        }
        res.at(pos.position()) = on;
    }
}

void dilate_dispatch(PyArrayObject* res, PyArrayObject* array, PyArrayObject* Bc) {
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
    dilate<type>(numpy_aligned_array<type>(res),numpy_array<type>(array),numpy_aligned_array<type>(Bc));\

        HANDLE_INTEGER_TYPES();
#undef HANDLE
    }
}


PyArrayObject* dilate(PyArrayObject* array, PyArrayObject* Bc) {
    PyArrayObject* res_a = (PyArrayObject*)PyArray_FromDims(array->nd,array->dimensions,PyArray_TYPE(array));
    if (!res_a) { 
        return NULL;
    }
    dilate_dispatch(res_a,array,Bc); 
    return res_a;
}
