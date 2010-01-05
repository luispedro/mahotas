#include <algorithm>
#include <queue>
#include <vector>
#include <cstdio>
#include <limits>
#include <iostream>

#include "numpypp/array.hpp"
#include "numpypp/dispatch.hpp"

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] = 
    "Type not understood. "
    "This is caused by either a direct call to _morph (which is dangerous: types are not checked!) or a bug in morph.py.\n";


template <typename T>
numpy::position central_position(const numpy::array_base<T>& array) {
    numpy::position res(array.raw_dims(), array.ndims());
    for (int i = 0, nd = array.ndims(); i != nd; ++i) res.position_[i] /= 2;
    return res;
}

template <typename T>
std::vector<numpy::position> neighbours(const numpy::aligned_array<T>& Bc, bool include_centre = false) {
    numpy::position centre = central_position(Bc);
    const unsigned N = Bc.size();
    typename numpy::aligned_array<T>::const_iterator startc = Bc.begin();
    std::vector<numpy::position> res;
    for (unsigned i = 0; i != N; ++i, ++startc) {
        if (!*startc) continue;
        if (startc.position() != centre || include_centre) {
            res.push_back(startc.position() - centre);
        }
    }
    return res;
}

template<typename T>
void erode(numpy::aligned_array<T> res, numpy::array<T> array, numpy::aligned_array<T> Bc) {
    const unsigned N = res.size();
    const unsigned N2 = Bc.size();
    const numpy::position centre = central_position(Bc);
    typename numpy::aligned_array<T>::iterator rpos = res.begin();

    for (int i = 0; i != N; ++i, ++rpos) {
        bool on = true;
        typename numpy::aligned_array<T>::iterator startc = Bc.begin();
        for (int j = 0; j != N2; ++j, ++startc) {
            if (*startc) {
                numpy::position npos = rpos.position() + startc.position() - centre;
                if (array.validposition(npos) && !array.at(npos)) {
                    on = false;
                    break;
                }
            }
        }
        *rpos = on;
    }
}


PyObject* py_erode(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* Bc;
    if (!PyArg_ParseTuple(args,"OO", &array, &Bc)) return NULL;
    PyArrayObject* res_a = (PyArrayObject*)PyArray_SimpleNew(array->nd,array->dimensions,PyArray_TYPE(array));
    if (!res_a) return NULL;
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
    erode<type>(numpy::aligned_array<type>(res_a),numpy::array<type>(array),numpy::aligned_array<type>(Bc));\

        HANDLE_INTEGER_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    return PyArray_Return(res_a);
}

template<typename T>
void dilate(numpy::aligned_array<T> res, numpy::array<T> array, numpy::aligned_array<T> Bc) {
    const unsigned N = res.size();
    const unsigned N2 = Bc.size();
    const numpy::position centre = central_position(Bc);

    typename numpy::array<T>::iterator pos = array.begin();
    for (int i = 0; i != N; ++i, ++pos) {
        if (*pos) {
            typename numpy::aligned_array<T>::iterator startc = Bc.begin();
            for (int j = 0; j != N2; ++j, ++startc) {
                if (*startc) {
                    numpy::position npos = pos.position() + startc.position() - centre;
                    if (res.validposition(npos)) {
                        res.at(npos) = *pos+*startc;
                    }
                }
            }
        }
    }

}

PyObject* py_dilate(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* Bc;
    if (!PyArg_ParseTuple(args,"OO", &array, &Bc)) return NULL;
    PyArrayObject* res_a = (PyArrayObject*)PyArray_SimpleNew(array->nd,array->dimensions,PyArray_TYPE(array));
    if (!res_a) return NULL;
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
    dilate<type>(numpy::aligned_array<type>(res_a),numpy::array<type>(array),numpy::aligned_array<type>(Bc));\

        HANDLE_INTEGER_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    return PyArray_Return(res_a);
}

void close_holes(numpy::aligned_array<bool> ref, numpy::aligned_array<bool> f, numpy::aligned_array<bool> Bc) {
    std::fill_n(f.data(),f. size(), false);

    std::vector<numpy::position> stack;
    const unsigned N = ref.size();
    const std::vector<numpy::position> Bc_neighbours = neighbours(Bc);
    const unsigned N2 = Bc_neighbours.size();
    for (int d = 0; d != ref.ndims(); ++d) {
        if (ref.dim(d) == 0) continue;
        numpy::position pos;
        pos.nd_ = ref.ndims();
        for (int di = 0; di != ref.ndims(); ++di) pos.position_[di] = 0;

        for (int i = 0; i != N/ref.dim(d); ++i) {
            pos.position_[d] = 0;
            if (!ref.at(pos) && !f.at(pos)) {
                f.at(pos) = true;
                stack.push_back(pos);
            }
            pos.position_[d] = ref.dim(d) - 1;
            if (!ref.at(pos) && !f.at(pos)) {
                f.at(pos) = true;
                stack.push_back(pos);
            }

            for (int j = 0; j != ref.ndims() - 1; ++j) {
                if (j == d) ++j;
                if (pos.position_[j] < ref.dim(j)) {
                    ++pos.position_[j];
                    break;
                }
                pos.position_[j] = 0;
            }
        }
    }
    while (!stack.empty()) {
        numpy::position pos = stack.back();
        stack.pop_back();
        std::vector<numpy::position>::const_iterator startc = Bc_neighbours.begin();
        for (int j = 0; j != N2; ++j, ++startc) {
            numpy::position npos = pos + *startc;
            if (ref.validposition(npos) && !ref.at(npos) && !f.at(npos)) {
                f.at(npos) = true;
                stack.push_back(npos);
            }
        }
    }
    for (bool* start = f.data(), *past = f.data() + f.size(); start != past; ++ start) {
        *start = !*start;
    }
}

PyObject* py_close_holes(PyObject* self, PyObject* args) {
    PyArrayObject* ref;
    PyArrayObject* Bc;
    if (!PyArg_ParseTuple(args,"OO", &ref, &Bc)) {
        return NULL;
    }
    PyArrayObject* res_a = (PyArrayObject*)PyArray_SimpleNew(ref->nd,ref->dimensions,PyArray_TYPE(ref));
    if (!res_a) return NULL;
    if (PyArray_TYPE(ref) != NPY_BOOL || PyArray_TYPE(Bc) != NPY_BOOL) {
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    close_holes(numpy::aligned_array<bool>(ref), numpy::aligned_array<bool>(res_a), numpy::aligned_array<bool>(Bc));
    return PyArray_Return(res_a);
}

struct MarkerInfo { 
    int cost;
    int idx;
    numpy::position pos;
    MarkerInfo(int cost, int idx, numpy::position pos)
        :cost(cost),
        idx(idx),
        pos(pos) {
        }
    bool operator < (const MarkerInfo& other) const {
        // We want the smaller argument to compare higher, so we reverse the order here:
        if (cost == other.cost) return idx > other.idx;
        return cost > other.cost;
    }
};

template<typename BaseType>
void cwatershed(numpy::aligned_array<BaseType> res, numpy::aligned_array<bool>* lines, numpy::array<BaseType> array, numpy::array<BaseType> markers, numpy::aligned_array<BaseType> Bc) {
    const unsigned N = res.size();
    const unsigned N2 = Bc.size();
    const numpy::position centre = central_position(Bc);
    int idx = 0;

    numpy::aligned_array<BaseType> cost = array_like(array);
    Py_XDECREF(cost.raw_array());
    std::fill_n(cost.data(),cost.size(),std::numeric_limits<BaseType>::max());
    numpy::aligned_array<bool> status((PyArrayObject*)PyArray_SimpleNew(array.ndims(),const_cast<npy_intp*>(array.raw_dims()),NPY_BOOL));
    Py_XDECREF(status.raw_array());
    std::fill_n(status.data(),status.size(),false);
    std::priority_queue<MarkerInfo> hqueue;
     
    typename numpy::array<BaseType>::iterator mpos = markers.begin();
    for (int i =0; i != N; ++i, ++mpos) {
        if (*mpos) {
            assert(markers.validposition(mpos.position()));
            hqueue.push(MarkerInfo(array.at(mpos.position()),idx++,mpos.position()));
            res.at(mpos.position()) = *mpos;
            cost.at(mpos.position()) = array.at(mpos.position());
        }
    }

    while (!hqueue.empty()) {
        const numpy::position pos = hqueue.top().pos;
        hqueue.pop();
        status.at(pos) = true;
        typename numpy::aligned_array<BaseType>::iterator Bi = Bc.begin();
        for (int j = 0; j != N2; ++j, ++Bi) {
            if (*Bi) {
                numpy::position npos = pos + Bi.position() - centre;
                if (status.validposition(npos)) {
                    if (!status.at(npos)) {
                        BaseType ncost = array.at(npos);
                        if (ncost < cost.at(npos)) {
                            cost.at(npos) = ncost;
                            res.at(npos) = res.at(pos);
                            hqueue.push(MarkerInfo(ncost,idx++,npos));
                        }
                    } else if (lines && res.at(pos) != res.at(npos) && !lines->at(npos)) {
                        lines->at(pos) = true;
                    }
                }
            }
        }
    }
}

PyObject* py_cwatershed(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* markers;
    PyArrayObject* Bc;
    int return_lines;
    if (!PyArg_ParseTuple(args,"OOOi", &array, &markers, &Bc,&return_lines)) {
        return NULL;
    }
    PyArrayObject* res_a = (PyArrayObject*)PyArray_SimpleNew(array->nd,array->dimensions,PyArray_TYPE(array));
    if (!res_a) return NULL;
    PyArrayObject* lines =  0;
    numpy::aligned_array<bool>* lines_a = 0;
    if (return_lines) {
        lines = (PyArrayObject*)PyArray_SimpleNew(array->nd,array->dimensions,PyArray_TYPE(array));
        if (!lines) return NULL;
        lines_a = new numpy::aligned_array<bool>(lines);
    }
    switch(PyArray_TYPE(array)) {
#define HANDLE(type) \
    cwatershed<type>(numpy::aligned_array<type>(res_a),lines_a,numpy::array<type>(array),numpy::array<type>(markers),numpy::aligned_array<type>(Bc));
        HANDLE_INTEGER_TYPES();
#undef HANDLE
        default:
        PyErr_SetString(PyExc_RuntimeError,TypeErrorMsg);
        return NULL;
    }
    if (return_lines) {
        delete lines_a;
        PyObject* ret_val = PyTuple_New(2);
        PyTuple_SetItem(ret_val,0,(PyObject*)res_a);
        PyTuple_SetItem(ret_val,1,(PyObject*)lines);
        return ret_val;
    }
    return PyArray_Return(res_a);
}


PyMethodDef methods[] = {
  {"dilate",(PyCFunction)py_dilate, METH_VARARGS, NULL},
  {"erode",(PyCFunction)py_erode, METH_VARARGS, NULL},
  {"cwatershed",(PyCFunction)py_cwatershed, METH_VARARGS, NULL},
  {"close_holes",(PyCFunction)py_close_holes, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_morph()
  {
    import_array();
    (void)Py_InitModule("_morph", methods);
  }

