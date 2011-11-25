typedef unsigned char uchar;
typedef unsigned short ushort;
#define HANDLE_INTEGER_TYPES() \
    case NPY_BOOL: HANDLE(bool); break; \
    case NPY_UBYTE: HANDLE(unsigned char); break; \
    case NPY_BYTE: HANDLE(char); break; \
    case NPY_SHORT: HANDLE(short); break; \
    case NPY_USHORT: HANDLE(unsigned short); break; \
    case NPY_INT: HANDLE(int); break; \
    case NPY_UINT: HANDLE(unsigned int); break; \
    case NPY_LONG: HANDLE(npy_long); break; \
    case NPY_ULONG: HANDLE(npy_ulong); break;

#define HANDLE_FLOAT_TYPES() \
    case NPY_FLOAT: HANDLE(float); break; \
    case NPY_DOUBLE: HANDLE(double); break;

#define HANDLE_TYPES() \
    HANDLE_INTEGER_TYPES() \
    HANDLE_FLOAT_TYPES()

#define SWITCH_ON_TYPES_OF(array) \
    switch(PyArray_TYPE(array)) { \
            HANDLE_TYPES();\
            default: \
            PyErr_SetString(PyExc_RuntimeError, "Dispatch on types failed!"); \
            return NULL; \
    }

#define SAFE_SWITCH_ON_TYPES_OF(array,return_null) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(return_null)

#define SAFE_SWITCH_ON_INTEGER_TYPES_OF(array,return_null) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_INTEGER_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(return_null)

#define SAFE_SWITCH_ON_FLOAT_TYPES_OF(array,return_null) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_FLOAT_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS(return_null)


#define CATCH_PYTHON_EXCEPTIONS(return_null) \
    catch (const PythonException& pe) { \
        PyErr_SetString(pe.type(), pe.message()); \
        if (return_null) return NULL; \
    } catch (const std::bad_alloc&) {\
        PyErr_NoMemory(); \
        if (return_null) return NULL; \
    }

