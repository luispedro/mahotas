// Copyright (C) 2008-2012  Luis Pedro Coelho <luis@luispedro.org>
//
// License: MIT (see COPYING file)

#include <algorithm>
#include <vector>
#include "numpypp/array.hpp"
#include "utils.hpp"

namespace {
struct Point {
    Point(int y_, int x_):y(y_), x(x_) { }
	long y, x;
};

inline
bool forward_cmp(const Point& a, const Point& b) {
	if (a.y == b.y) return a.x < b.x;
	return a.y < b.y;
}
inline
bool reverse_cmp(const Point& a, const Point& b) {
	if (a.y == b.y) return a.x > b.x;
	return a.y > b.y;
}

inline
double isLeft(Point p0, Point p1, Point p2) {
	return (p1.y-p0.y)*(p2.x-p0.x) - (p2.y-p0.y)*(p1.x-p0.x);
}

unsigned inPlaceScan(Point* P, unsigned N, bool reverse) {
	if (reverse) {
		std::sort(P, P+N, reverse_cmp);
	} else {
		std::sort(P, P+N, forward_cmp);
	}
	int h = 1;
	for (int i = 1; i != int(N); ++i) {
		while (h >= 2 && isLeft(P[h-2],P[h-1],P[i]) >= 0) {
			--h;
		}
		std::swap(P[h],P[i]);
		++h;
	}
	return h;
}

unsigned inPlaceGraham(std::vector<Point>& Pv) {
    const int N = Pv.size();
    if (N <= 3) return N;
    Point* P = &Pv[0];
	int h = inPlaceScan(P,N,false);
	for (int i = 0; i != h - 1; ++i) {
		std::swap(P[i],P[i+1]);
	}
	int h_=inPlaceScan(P+h-2,N-h+2,true);
	return h + h_ - 2;
}

PyObject*
convexhull(PyObject* self, PyObject* args) {
	PyArrayObject* array;
	if (!PyArg_ParseTuple(args,"O", &array) ||
           !PyArray_ISCARRAY(array) ||
           !PyArray_EquivTypenums(PyArray_TYPE(array), NPY_BOOL)) return 0;

    holdref r(array);
	unsigned h;
    std::vector<Point> Pv;
    try { // Release GIL
        gil_release nogil;
        const numpy::aligned_array<bool> barray(array);
        const int N0 = barray.dim(0);
        const int N1 = barray.dim(1);
        for (int y = 0; y != N0; ++y) {
            for (int x = 0; x != N1; ++x) {
                if (barray.at(y,x)) Pv.push_back(Point(y,x));
            }
        }
        h = inPlaceGraham(Pv);
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return NULL;
    }
    npy_intp dims[2];
    dims[0] = h;
    dims[1] = 2;
    PyObject* output = PyArray_SimpleNew(2, dims, NPY_INTP);
	if (!output) {
		PyErr_NoMemory();
		return 0;
	}
    npy_intp* oiter = numpy::ndarray_cast<npy_intp*>(output);
	for (unsigned i = 0; i != h; ++i) {
        *oiter++ = Pv[i].y;
        *oiter++ = Pv[i].x;
	}
	return output;
}


}

PyMethodDef methods[] = {
  {"convexhull", convexhull, METH_VARARGS , "compute convex hull"},
  {NULL, NULL,0,NULL},
};

DECLARE_MODULE(_convex)
