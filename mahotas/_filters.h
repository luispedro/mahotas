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

int init_filter_offsets(PyArrayObject *array, bool *footprint,
         const npy_intp * const fshape, npy_intp* origins,
         const ExtendMode mode, npy_intp **offsets, npy_intp *border_flag_value,
         npy_intp **coordinate_offsets);

template <typename T>
struct filter_iterator {
    /* Move to the next point in an array, possible changing the pointer
         to the filter offsets when moving into a different region in the
         array: */
    filter_iterator(PyArrayObject* array, PyArrayObject* filter)
        :offsets_(0)
        ,coordinate_offsets_(0)
    {
        init_filter_offsets(array, 0, PyArray_DIMS(array), 0,
        EXTEND_NEAREST, &offsets_, &border_flag_value_, 0);

    }
    ~filter_iterator() {
        delete [] offsets_;
        if (coordinate_offsets_) delete coordinate_offsets_;
    }
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
        npy_intp* offsets_;
        npy_intp* coordinate_offsets_;
        npy_intp strides_[NPY_MAXDIMS];
        npy_intp minbound_[NPY_MAXDIMS];
        npy_intp maxbound_[NPY_MAXDIMS];
        npy_intp border_flag_value_;
};

