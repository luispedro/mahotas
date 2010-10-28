/* The different boundary conditions. The mirror condition is not used
     by the python code, but C code is kept around in case we might wish
     to add it. */
typedef enum {
    NI_EXTEND_FIRST = 0,
    NI_EXTEND_NEAREST = 0,
    NI_EXTEND_WRAP = 1,
    NI_EXTEND_REFLECT = 2,
    NI_EXTEND_MIRROR = 3,
    NI_EXTEND_CONSTANT = 4,
    NI_EXTEND_LAST = NI_EXTEND_CONSTANT,
    NI_EXTEND_DEFAULT = NI_EXTEND_MIRROR
} NI_ExtendMode;
