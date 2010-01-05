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
