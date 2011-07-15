// Copyright (C) 2008-2010  Murphy Lab
// Written by Luis Pedro Coelho <luis@luispedro.org>
// Carnegie Mellon University
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published
// by the Free Software Foundation; either version 2 of the License,
// or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
// 02110-1301, USA.

#include <algorithm>
#include <vector>
#include "utils.hpp"
#include <Python.h>

namespace {
struct Point {
	long x, y;
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
		std::sort(P,P+N,reverse_cmp);
	} else {
		std::sort(P,P+N,forward_cmp);
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

unsigned inPlaceGraham(Point* P, unsigned N) {
	int h = inPlaceScan(P,N,false);
	for (int i = 0; i != h - 1; ++i) {
		std::swap(P[i],P[i+1]);
	}
	int h_=inPlaceScan(P+h-2,N-h+2,true);
	return h + h_ - 2;
}

PyObject*
convexhull(PyObject* self, PyObject* args) {
	PyObject* input;
	if (!PyArg_ParseTuple(args,"O",&input)) return 0;
	const Py_ssize_t N = PyList_Size(input);
    if (N <= 3) {
        // The python wrapper should have stopped this from happening
        Py_RETURN_NONE;
        //    Py_INCREF(input);
        //    return input;
    }
    std::vector<Point> Pv;
    Pv.resize(N);
    Point* P = &Pv[0];
	for (Py_ssize_t i = 0; i != N; ++i) {
		PyObject* tup = PyList_GetItem(input,i);
		long y = PyInt_AsLong(PyTuple_GetItem(tup,0));
		long x = PyInt_AsLong(PyTuple_GetItem(tup,1));
		P[i].y=y;
		P[i].x=x;
	}
	unsigned h;
    { // Release GIL
        gil_release nogil;
        h = inPlaceGraham(P,N);
    }
	PyObject* output = PyList_New(h);
	if (!output) {
		PyErr_NoMemory();
		return 0;
	}
	for (unsigned i = 0; i != h; ++i) {
		PyObject* tup = PyTuple_New(2);
        if (!tup) {
            PyErr_NoMemory();
            return 0;
        }
		PyTuple_SetItem(tup,0,PyInt_FromLong(P[i].y));
		PyTuple_SetItem(tup,1,PyInt_FromLong(P[i].x));
		PyList_SetItem(output,i,tup);
	}
	return output;
}


}

PyMethodDef methods[] = {
  {"convexhull", convexhull, METH_VARARGS , "compute convex hull"},
  {NULL, NULL,0,NULL},
};

extern "C"
void init_convex()
  {
    (void)Py_InitModule("_convex", methods);
  }

