// Part of mahotas. See LICENSE file for License
// Copyright 2008-2010 Luis Pedro Coelho <luis@luispedro.org>

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}


struct holdref {
    holdref(PyObject* obj, bool incref=true)
        :obj(obj) {
        if (incref) Py_XINCREF(obj);
    }
    holdref(PyArrayObject* obj, bool incref=true)
        :obj((PyObject*)obj) {
        if (incref) Py_XINCREF(obj);
    }
    ~holdref() { Py_XDECREF(obj); }
  
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


// This encapsulates the arguments to PyErr_SetString
// The reason that it doesn't call PyErr_SetString directly is that we wish
// that this be throw-able in an environment where the thread might not own the
// GIL as long as it is caught when the GIL is held.
struct PythonException {
    PythonException(PyObject *type, const char *message)
        :type_(type)
        ,message_(message)
        { }

    PyObject* type() const { return type_; }
    const char* message() const { return message_; }

    PyObject* const type_;
    const char* const message_;
};

