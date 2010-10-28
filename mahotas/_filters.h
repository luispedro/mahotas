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

/* Move to the next point in two arrays, possible changing the pointer
     to the filter offsets when moving into a different region in the
     array: */
#define NI_FILTER_NEXT2(iteratorf, iterator1, iterator2,    \
                                                pointerf, pointer1, pointer2)       \
{                                                           \
    int _ii;                                                  \
    for(_ii = (iterator1).rank_m1; _ii >= 0; _ii--) {         \
        npy_intp _pp = (iterator1).coordinates[_ii];        \
        if (_pp < (iterator1).dimensions[_ii]) {                \
            if (_pp < (iteratorf).bound1[_ii] ||                  \
                                                        _pp >= (iteratorf).bound2[_ii]) \
                pointerf += (iteratorf).strides[_ii];               \
            (iterator1).coordinates[_ii]++;                       \
            pointer1 += (iterator1).strides[_ii];                 \
            pointer2 += (iterator2).strides[_ii];                 \
            break;                                                \
        } else {                                                \
            (iterator1).coordinates[_ii] = 0;                     \
            pointer1 -= (iterator1).backstrides[_ii];             \
            pointer2 -= (iterator2).backstrides[_ii];             \
            pointerf -= (iteratorf).backstrides[_ii];             \
        }                                                       \
    }                                                         \
}
