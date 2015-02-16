#ifndef MAHOTAS_FILTER_H_INCLUDE_GUARD_
#define MAHOTAS_FILTER_H_INCLUDE_GUARD_
// Copyright (C) 2003-2005 Peter J. Verveer
// Copyright (C) 2010-2013 Luis Pedro Coelho
// LICENSE: MIT

#include <vector>
#include <cassert>
#include <limits>
#include "numpypp/array.hpp"


/* The different boundary conditions. The mirror condition is not used
     by the python code, but C code is kept around in case we might wish
     to add it. */
typedef enum {
    ExtendNearest = 0,
    ExtendWrap = 1,
    ExtendReflect = 2,
    ExtendMirror = 3,
    ExtendConstant = 4,
    ExtendIgnore = 5
} ExtendMode;
const ExtendMode ExtendLast = ExtendIgnore;

const npy_intp border_flag_value = std::numeric_limits<npy_intp>::max();

inline
npy_intp fix_offset(const ExtendMode mode, npy_intp cc, const npy_intp len) {
    /* apply boundary conditions, if necessary: */
    switch (mode) {
    case ExtendMirror:
        if (cc < 0) {
            if (len <= 1) {
                return 0;
            } else {
                int sz2 = 2 * len - 2;
                cc = sz2 * (int)(-cc / sz2) + cc;
                return cc <= 1 - len ? cc + sz2 : -cc;
            }
        } else if (cc >= len) {
            if (len <= 1) {
                return 0;
            } else {
                int sz2 = 2 * len - 2;
                cc -= sz2 * (int)(cc / sz2);
                if (cc >= len)
                    cc = sz2 - cc;
            }
        }
        return cc;

    case ExtendReflect:
        if (cc < 0) {
            if (len <= 1) {
                return 0;
            } else {
                int sz2 = 2 * len;
                if (cc < -sz2)
                    cc = sz2 * (int)(-cc / sz2) + cc;
                cc = cc < -len ? cc + sz2 : -cc - 1;
            }
        } else if (cc >= len) {
            if (len <= 1) {
                return 0;
            } else {
                int sz2 = 2 * len;
                cc -= sz2 * (int)(cc / sz2);
                if (cc >= len)
                    cc = sz2 - cc - 1;
            }
        }
        return cc;
    case ExtendWrap:
        if (cc < 0) {
            if (len <= 1) {
                return 0;
            } else {
                int sz = len;
                cc += sz * (int)(-cc / sz);
                if (cc < 0)
                    cc += sz;
            }
        } else if (cc >= len) {
            if (len <= 1) {
                return 0;
            } else {
                int sz = len;
                cc -= sz * (int)(cc / sz);
            }
        }
        return cc;
    case ExtendNearest:
        if (cc < 0) return 0;
        if (cc >= len) return len - 1;
        return cc;
    case ExtendIgnore:
    case ExtendConstant:
        if (cc < 0 || cc >= len)
            return border_flag_value;
        return cc;
    }
    assert(false); // We should never get here
    return 0;
}


int init_filter_offsets(PyArrayObject *array, bool *footprint,
         const npy_intp * const fshape, npy_intp* origins,
         const ExtendMode mode, std::vector<npy_intp>& offsets,
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
    filter_iterator(PyArrayObject* array, PyArrayObject* filter, ExtendMode mode=ExtendNearest, bool compress=true)
        :filter_data_(numpy::ndarray_cast<T*>(filter))
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
                footprint[i] = !!(*fiter);
            }
        }
        size_ = init_filter_offsets(array, footprint, PyArray_DIMS(filter), 0,
                    mode, offsets_, 0);
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

        init_filter_iterator(PyArray_NDIM(filter), PyArray_DIMS(filter), size_,
            PyArray_DIMS(array), /*origins*/0,
            this->strides_, this->backstrides_,
            this->minbound_, this->maxbound_);
        cur_offsets_idx_ = this->offsets_.begin();
    }
    ~filter_iterator() {
        if (own_filter_data_) delete [] filter_data_;
    }
    template <typename OtherIterator>
    void iterate_both(OtherIterator& iterator) {
        for (int d = 0; d < nd_; ++d) {
            const npy_intp p = iterator.index_rev(d);
            if (p < (iterator.dimension_rev(d) - 1)) {
                if (p < this->minbound_[d] || p >= this->maxbound_[d]) {
                    this->cur_offsets_idx_ += this->strides_[d];
                }
                break;
            }
            this->cur_offsets_idx_ -= this->backstrides_[d];
            assert(this->cur_offsets_idx_ >= this->offsets_.begin());
            assert(this->cur_offsets_idx_ < this->offsets_.end());
        }
        ++iterator;
    }

    template <typename OtherIterator>
    bool retrieve(const OtherIterator& iterator, const npy_intp j, T& array_val) {
        assert((j >= 0) && (j < size_));
        if (this->cur_offsets_idx_[j] == border_flag_value) return false;
        array_val = *( (&*iterator) + this->cur_offsets_idx_[j]);
        return true;
    }
    template <typename OtherIterator>
    void set(const OtherIterator& iterator, npy_intp j, const T& val) {
        assert(this->cur_offsets_idx_[j] != border_flag_value);
        *( (&*iterator) + this->cur_offsets_idx_[j]) = val;
    }

    const T& operator [] (const npy_intp j) const { assert(j < size_); return filter_data_[j]; }
    npy_intp size() const { return size_; }
    private:
        const T* filter_data_;
        bool own_filter_data_;
        std::vector<npy_intp>::const_iterator cur_offsets_idx_;
        npy_intp size_;
        const npy_intp nd_;
        std::vector<npy_intp> offsets_;
        npy_intp strides_[NPY_MAXDIMS];
        npy_intp backstrides_[NPY_MAXDIMS];
        npy_intp minbound_[NPY_MAXDIMS];
        npy_intp maxbound_[NPY_MAXDIMS];
};

#endif // MAHOTAS_FILTER_H_INCLUDE_GUARD_
