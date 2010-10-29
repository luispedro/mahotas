extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}


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

template <typename T>
struct filter_iterator {
    /* Move to the next point in an array, possible changing the pointer
         to the filter offsets when moving into a different region in the
         array: */
    template <typename OtherIterator>
    void iterate_with(OtherIterator& iterator) {
        for (int i = 0; i != iterator.position_.nd_; ++i) {
            npy_intp& p = iterator.position_.position_[i];
            iterator.data += iterator.steps_[i];
            ++p;
            if (p < this->minbound_[i] || p >= this->maxbound_[i]) {
                this->data_ += this->strides_[i];
            }
            if (p != iterator.dimension_[i]) {
                return;
            }
            iterator.position_.position_[i] = 0;
        }
    }
    private:
        T* data_;
        npy_intp strides_[NPY_MAXDIMS];
        npy_intp minbound_[NPY_MAXDIMS];
        npy_intp maxbound_[NPY_MAXDIMS];
};

