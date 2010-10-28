/* The different boundary conditions. The mirror condition is not used
     by the python code, but C code is kept around in case we might wish
     to add it. */
typedef enum {
    EXTEND_FIRST = 0,
    EXTEND_NEAREST = 0,
    EXTEND_WRAP = 1,
    EXTEND_REFLECT = 2,
    EXTEND_MIRROR = 3,
    EXTEND_CONSTANT = 4,
    EXTEND_LAST = EXTEND_CONSTANT,
    EXTEND_DEFAULT = EXTEND_MIRROR
} ExtendMode;
