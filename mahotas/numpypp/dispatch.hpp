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

#define SAFE_SWITCH_ON_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS

#define SAFE_SWITCH_ON_INTEGER_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_INTEGER_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS

#define SAFE_SWITCH_ON_FLOAT_TYPES_OF(array) \
    try { \
        switch(PyArray_TYPE(array)) { \
                HANDLE_FLOAT_TYPES();\
                default: \
                PyErr_SetString(PyExc_RuntimeError, "Dispatch on types failed!"); \
                return NULL; \
        } \
    } \
    CATCH_PYTHON_EXCEPTIONS


#define CATCH_PYTHON_EXCEPTIONS \
    catch (const PythonException& pe) { \
        PyErr_SetString(pe.type(), pe.message()); \
        return NULL; \
    } catch (const std::bad_alloc&) {\
        PyErr_NoMemory(); \
        return NULL; \
    }

