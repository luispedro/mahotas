// Part of mahotas. See LICENSE file for License
// Copyright 2008-2010 Luis Pedro Coelho <lpc@cmu.edu>

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}


struct holdref {
    holdref(PyObject* obj, bool incref=true)
        :obj(obj) {
        if (incref) Py_INCREF(obj);
    }
    holdref(PyArrayObject* obj, bool incref=true)
        :obj((PyObject*)obj) {
        if (incref) Py_INCREF(obj);
    }
    ~holdref() { Py_DECREF(obj); }
  
private:  
    PyObject* const obj;
};

struct gil_release {
    gil_release() {
        _save = PyEval_SaveThread();
        active_ = true;
    }
    ~gil_release() {
        if (active_) restore();
    }
    void restore() {
        PyEval_RestoreThread(_save);
        active_ = false;
    }
    PyThreadState *_save;
    bool active_;
};


