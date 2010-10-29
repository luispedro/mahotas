#include <cstring>

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

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
 
const bool boolvals[] =    { true, true, true, false, false, false };
// edge
const npy_intp edelta0[] = {   -1,   -1,   -1,    +1,    +1,    +1 };
const npy_intp edelta1[] = {   -1,    0,   +1,    -1,     0,    +1 };
// corner
const npy_intp cdelta0[] = {   -1,   -1,    0,     0,    +1,    +1 };
const npy_intp cdelta1[] = {   -1,   +1,   -1,    +1,     0,    +1 };

void fill_data(PyArrayObject* array, bool* data, npy_intp* offset, const bool flip, const npy_intp* delta0, const npy_intp* delta1) {    
    for (int j = 0; j != Element_Size; ++j) {
        data[j] = (flip ? ! boolvals[j]: boolvals[j]);
        offset[j] = coordinates_delta(array, delta0[j], delta1[j]);
    }
}


PyObject* py_thin(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    PyArrayObject* buffer;
    PyArrayObject* previous;
    if (!PyArg_ParseTuple(args,"OOO", &array, &buffer, &previous)) return NULL;


    const int Nr_Elements = 8;
    structure_element elems[Nr_Elements];
    fill_data(array, elems[0].data, elems[0].offset, false, edelta0, edelta1);
    fill_data(array, elems[1].data, elems[1].offset, false, cdelta0, cdelta1);
    fill_data(array, elems[2].data, elems[2].offset, false, edelta1, edelta0);
    fill_data(array, elems[3].data, elems[3].offset, false, cdelta1, cdelta0);
    fill_data(array, elems[4].data, elems[4].offset,  true, edelta0, edelta1);
    fill_data(array, elems[5].data, elems[5].offset,  true, cdelta0, cdelta1);
    fill_data(array, elems[6].data, elems[6].offset,  true, edelta1, edelta0);
    fill_data(array, elems[7].data, elems[7].offset,  true, cdelta1, cdelta0);

    PyArray_FILLWBYTE(buffer, 0);
    const npy_int N = PyArray_SIZE(array);
    do {
        std::memcpy(PyArray_DATA(previous), PyArray_DATA(array), PyArray_NBYTES(array));
        for (int i = 0; i != Nr_Elements; ++i) {
            fast_hitmiss(array, elems[i], buffer);
            bool* pa = reinterpret_cast<bool*>(PyArray_DATA(array));
            const bool* pb = reinterpret_cast<bool*>(PyArray_DATA(buffer));
            for (int j = 0; j != N; ++j) {
                if (*pb) {
                    *pa = false;
                }
                ++pa;
                ++pb;
            }
        }
    } while (std::memcmp(PyArray_DATA(previous), PyArray_DATA(array), PyArray_NBYTES(array)));

    Py_INCREF(array);
    return PyArray_Return(array);
}


PyMethodDef methods[] = {
  {"thin",(PyCFunction)py_thin, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_thin()
  {
    import_array();
    (void)Py_InitModule("_thin", methods);
  }

