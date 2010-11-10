extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

#include <iostream>

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
        :filter_data_(reinterpret_cast<const T* const>(PyArray_DATA(filter)))
        ,offsets_(0)
        ,coordinate_offsets_(0)
    {
        size_ = init_filter_offsets(array, 0, PyArray_DIMS(filter), 0,
                    EXTEND_NEAREST, &offsets_, &border_flag_value_, 0);
        cur_offsets_ = offsets_;
        nd_ = PyArray_NDIM(array);
        //init_filter_boundaries(array, filter, minbound_, maxbound_);
        for (int i = 0; i != PyArray_NDIM(filter); ++i) {
            minbound_[i] = 1;
            maxbound_[i] = PyArray_DIM(array,i) - 2;
        }
        this->strides_[this->nd_ - 1] = size_;
        for (int i = nd_ - 2; i >= 0; --i) {
            const npy_intp step = std::min<npy_intp>(PyArray_DIM(filter, i + 1), PyArray_DIM(array, i + 1));
            this->strides_[i] = this->strides_[i + 1] * step;
        }
        for (int i = 0; i < this->nd_; ++i) {
            const npy_intp step = std::min<npy_intp>(PyArray_DIM(array, i), PyArray_DIM(filter, i));
            this->backstrides_[i] = (step - 1) * this->strides_[i];
        }
    }
    ~filter_iterator() {
        delete [] offsets_;
        if (coordinate_offsets_) delete coordinate_offsets_;
    }
    template <typename OtherIterator>
    void iterate_with(const OtherIterator& iterator) {
        for (int i = nd_ - 1; i >= 0; --i) {
            npy_intp p = iterator.index(i);
            if (p < (iterator.dimension(i) - 1)) {
                if (p < this->minbound_[i] || p >= this->maxbound_[i]) {
                    this->cur_offsets_ += this->strides_[i];
                }
                return;
            }
            this->cur_offsets_ -= this->backstrides_[i];
            assert( (this->cur_offsets_ - this->offsets_) >= 0);
        }
    }
    template <typename OtherIterator>
    void retrieve(const OtherIterator& iterator, const npy_intp j, T& array_val, T& filter_val) {
        array_val = *( (&*iterator) + this->cur_offsets_[j]);
        filter_val = filter_data_[j];
    }
    npy_intp size() const { return size_; }
    private:
        const T* const filter_data_;
        npy_intp* cur_offsets_;
        npy_intp size_;
        npy_intp nd_;
        npy_intp* offsets_;
        npy_intp* coordinate_offsets_;
        npy_intp strides_[NPY_MAXDIMS];
        npy_intp backstrides_[NPY_MAXDIMS];
        npy_intp minbound_[NPY_MAXDIMS];
        npy_intp maxbound_[NPY_MAXDIMS];
        npy_intp border_flag_value_;
};

