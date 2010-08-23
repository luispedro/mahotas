#include <complex>
#include <cmath>
extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace{

const char TypeErrorMsg[] =
    "Type not understood. "
    "This is caused by either a direct call to _zernike (which is dangerous: types are not checked!) or a bug in zernike.py.\n";

double _factorialtable[] = { 
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600
};

PyObject* py_znl(PyObject* self, PyObject* args) {
    using std::pow;
    using std::atan2;
    using std::polar;
    using std::conj;
    using std::complex;


    PyArrayObject* Xa;
    PyArrayObject* Ya;
    PyArrayObject* Pa;
    double n;
    double l;
    if (!PyArg_ParseTuple(args,"OOOdd", &Xa, &Ya, &Pa, &n, &l)) return NULL;
    double* X = static_cast<double*>(PyArray_DATA(Xa));
    double* Y = static_cast<double*>(PyArray_DATA(Ya));
    double* P = static_cast<double*>(PyArray_DATA(Pa));
    int Nelems = PyArray_SIZE(Xa);
    complex<double> Vnl = 0.0;
    complex<double> v = 0.;
    for (int i = 0; i != Nelems; ++i) {
        double x=X[i];
        double y=Y[i];
        double p=P[i];
        Vnl = 0.;
        for(int m = 0; m <= (n-l)/2; m++) {
            double f = (m & 1) ? -1 : 1;
            Vnl += f * _factorialtable[int(n-m)] /
                   ( _factorialtable[m] * _factorialtable[int((n - 2*m + l) / 2)] * _factorialtable[int((n - 2*m - l) / 2)] ) *
                   ( pow( sqrt(x*x + y*y), (double)(n - 2*m)) ) *
                   polar(1.0, l*atan2(y,x)) ;
        }
        v += p * conj(Vnl);
    }
    return PyComplex_FromDoubles(v.real(), v.imag());

}
PyMethodDef methods[] = {
  {"znl",(PyCFunction)py_znl, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_zernike()
  {
    import_array();
    (void)Py_InitModule("_zernike", methods);
  }

