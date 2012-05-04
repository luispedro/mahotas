#include <vector>
#include <assert.h>
#include "numpypp/array.hpp"

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

npy_intp fix_offset(const ExtendMode mode, npy_intp cc, const npy_intp len, const npy_intp border_flag_value);

int init_filter_offsets(PyArrayObject *array, bool *footprint,
         const npy_intp * const fshape, npy_intp* origins,
         const ExtendMode mode, std::vector<npy_intp>& offsets, npy_intp *border_flag_value,
         std::vector<npy_intp>* coordinate_offsets);
void init_filter_iterator(const int rank, const npy_intp *fshape,
                    const npy_intp filter_size, const npy_intp *ashape,
                    const npy_intp *origins,
                    npy_intp* strides, npy_intp* backstrides,
                    npy_intp* minbound, npy_intp* maxbound);

template <typename T>
struct filter_iterator {
    /* Move to the next point in an array, possible changing the pointer
         to the filter offsets when moving into a different region in the
         array: */
    filter_iterator(PyArrayObject* array, PyArrayObject* filter, ExtendMode mode = EXTEND_NEAREST, bool compress=true)
        :filter_data_(reinterpret_cast<const T* const>(PyArray_DATA(filter)))
        ,own_filter_data_(false)
        ,nd_(PyArray_NDIM(array))
    {
        numpy::aligned_array<T> filter_array(filter);
        const npy_intp filter_size = filter_array.size();
        bool* footprint = 0;
        if (compress) {
            footprint = new bool[filter_size];
            typename numpy::aligned_array<T>::iterator fiter = filter_array.begin();
            for (int i = 0; i != filter_size; ++i, ++fiter) {
                footprint[i] = bool(*fiter);
            }
        }
        size_ = init_filter_offsets(array, footprint, PyArray_DIMS(filter), 0,
                    mode, offsets_, &border_flag_value_, 0);
        if (compress) {
            int j = 0;
            T* new_filter_data = new T[size_];
            typename numpy::aligned_array<T>::iterator fiter = filter_array.begin();
            for (int i = 0; i != filter_size; ++i, ++fiter) {
                if (*fiter) {
                    new_filter_data[j++] = *fiter;
                }
            }
            filter_data_ = new_filter_data;
            own_filter_data_ = true;
            delete [] footprint;
        }

        cur_offsets_idx_ = 0;
        init_filter_iterator(PyArray_NDIM(filter), PyArray_DIMS(filter), size_,
            PyArray_DIMS(array), /*origins*/0,
            this->strides_, this->backstrides_,
            this->minbound_, this->maxbound_);
    }
    ~filter_iterator() {
        if (own_filter_data_) delete [] filter_data_;
    }
    template <typename OtherIterator>
    void iterate_both(OtherIterator& iterator) {
        for (int i = nd_ - 1; i >= 0; --i) {
            npy_intp p = iterator.index(i);
            if (p < (iterator.dimension(i) - 1)) {
                if (p < this->minbound_[i] || p >= this->maxbound_[i]) {
                    this->cur_offsets_idx_ += this->strides_[i];
                }
                break;
            }
            this->cur_offsets_idx_ -= this->backstrides_[i];
            assert( this->cur_offsets_idx_ >= 0 );
            assert( this->cur_offsets_idx_ < this->offsets_.size() );
        }
        ++iterator;
    }

    template <typename OtherIterator>
    bool retrieve(const OtherIterator& iterator, const npy_intp j, T& array_val) {
        if (this->offsets_[cur_offsets_idx_+j] == border_flag_value_) return false;
        assert((j >= 0) && (j < size_));
        array_val = *( (&*iterator) + this->offsets_[cur_offsets_idx_+j]);
        return true;
    }
    template <typename OtherIterator>
    void set(const OtherIterator& iterator, npy_intp j, const T& val) {
        assert(this->offsets_[cur_offsets_idx_+j] != border_flag_value_);
        *( (&*iterator) + this->offsets_[cur_offsets_idx_+j]) = val;
    }

    const T& operator [] (const npy_intp j) const { assert(j < size_); return filter_data_[j]; }
    npy_intp size() const { return size_; }
    private:
        const T* filter_data_;
        bool own_filter_data_;
        unsigned cur_offsets_idx_;
        npy_intp size_;
        const npy_intp nd_;
        std::vector<npy_intp> offsets_;
        npy_intp strides_[NPY_MAXDIMS];
        npy_intp backstrides_[NPY_MAXDIMS];
        npy_intp minbound_[NPY_MAXDIMS];
        npy_intp maxbound_[NPY_MAXDIMS];
        npy_intp border_flag_value_;
};

