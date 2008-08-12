typedef unsigned char uchar;
typedef unsigned short ushort;
#define NUMPYPP_CASE(typenum,type) \
    case typenum: HANDLE(type); break;
#define HANDLE_INTEGER_TYPES() \
    case NPY_UBYTE: HANDLE(uchar); break; \
    case NPY_BYTE: HANDLE(char); break; \
    case NPY_SHORT: HANDLE(short); break; \
    case NPY_USHORT: HANDLE(ushort); break; \
    case NPY_INT: HANDLE(int); break; \
    case NPY_UINT: HANDLE(unsigned); break;
