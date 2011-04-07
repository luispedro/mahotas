#ifndef MAHOTAS_NUMPYPP_ARRAY_HPP_INCLUDE_GUARD_LPC_
#define MAHOTAS_NUMPYPP_ARRAY_HPP_INCLUDE_GUARD_LPC_
/* Copyright 2008-2011 (C)
 * Luis Pedro Coelho <luis@luispedro.org>
 * License GPL Version 2, or later.
 */


#include <iterator>
#include <algorithm>
#include <cstring>
#include <ostream>
#include <iostream>
#include <cassert>

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

#include <stdio.h>

#include "numpy.hpp"

#ifndef __GNUC__
#define __PRETTY_FUNCTION__ ""
#endif

template <typename T>
struct filter_iterator;

namespace numpy {
typedef npy_intp index_type;
const unsigned index_type_number = NPY_INTP;

struct position {
    position()
        :nd_(0) {
        }
    position(const npy_intp* pos, int nd)
        :nd_(nd)
        { for (int i = 0; i != nd_; ++i) position_[i]=pos[i]; }
    npy_intp operator [] (unsigned pos) const { return this->position_[pos]; }
    int nd_;
    npy_intp position_[NPY_MAXDIMS];
    bool operator == (const position& other) { return !std::memcmp(this->position_,other.position_,sizeof(this->position_[0])*this->nd_); }
    bool operator != (const position& other) { return !(*this == other); }

    static position from1(npy_intp p0) {
        position res;
        res.nd_ = 1;
        res.position_[0] = p0;
        return res;
    }
    static position from2(npy_intp p0, npy_intp p1) {
        position res;
        res.nd_ = 2;
        res.position_[0] = p0;
        res.position_[1] = p1;
        return res;
    }
    static position from3(npy_intp p0, npy_intp p1, npy_intp p2) {
        position res;
        res.nd_ = 3;
        res.position_[0] = p0;
        res.position_[1] = p1;
        res.position_[2] = p2;
        return res;
    }
};

inline
position operator + (const position& a, const position& b) {
    assert(a.nd_ == b.nd_);
    position res = a;
    for (int i = 0; i != a.nd_; ++i) res.position_[i] += b.position_[i];
    return res;
}

inline
position operator - (const position& a, const position& b) {
    assert(a.nd_ == b.nd_);
    position res = a;
    for (int i = 0; i != a.nd_; ++i) res.position_[i] -= b.position_[i];
    return res;
}

inline
bool operator == (const position& a, const position& b) {
    if (a.nd_ != b.nd_) return false;
    for (int i = 0; i != a.nd_; ++i) if (a.position_[i] != b.position_[i]) return false;
    return true;
}

inline
bool operator != (const position& a, const position b) {
    return !(a == b);
}

template <typename T>
T& operator << (T& out, const numpy::position& p) {
    out << "[";
    for (int d = 0; d != p.nd_; ++d) {
        out << p.position_[d] << ":";
    }
    out << "]";
    return out;
}


template <typename BaseType>
struct iterator_base : std::iterator<std::forward_iterator_tag, BaseType>{
    friend class ::filter_iterator<BaseType>;
    protected:
        BaseType* data_;
        // steps is similar to strides, but more useful for iteration, see implementation of operator ++
        // Also, I divide by sizeof(BaseType)
        int steps_[NPY_MAXDIMS];
        int dimensions_[NPY_MAXDIMS];
        // This is not actually the position we are at, but the reverse of the position!
        ::numpy::position position_;

    public:
        iterator_base(PyArrayObject* array) {
            int nd = array->nd;
            position_.nd_=nd;
            data_=reinterpret_cast<BaseType*>(array->data);
            for (int i = 0; i != position_.nd_; ++i) position_.position_[i]=0;
            unsigned cummul = 0;
            for (int i = 0; i != position_.nd_; ++i) {
                dimensions_[i] = array->dimensions[nd-i-1];
                steps_[i] = array->strides[nd-i-1]/sizeof(BaseType)-cummul;
                cummul *= array->dimensions[nd-i-1];
                cummul += steps_[i]*array->dimensions[nd-i-1];
            }
        }

        iterator_base& operator ++ () {
            for (int i = 0; i != position_.nd_; ++i) {
                data_ += steps_[i];
                ++position_.position_[i];
                if (position_.position_[i] != dimensions_[i]) {
                    return *this;
                }
                position_.position_[i] = 0;
            }
            return *this;
        }

        int index(unsigned i) const { return index_rev(position_.nd_ - i - 1); }
        int index_rev(unsigned i) const { return position_.position_[i]; }
        npy_intp dimension(unsigned i) const { return dimension_rev(position_.nd_ - i - 1); }
        npy_intp dimension_rev(unsigned i) const { return dimensions_[i]; }

        bool operator == (const iterator_base& other) { return this->position_ == other.position_; }
        bool operator != (const iterator_base& other) { return !(*this == other); }

        ::numpy::position position() const {
            ::numpy::position res = position_;
            std::reverse(res.position_,res.position_+res.nd_);
            return res;
        }
        friend inline
        std::ostream& operator << (std::ostream& out, const iterator_base& iter) {
            return out << "I {" << iter.position_ << "}";
        }
};



template <typename BaseType>
class iterator_type : public iterator_base<BaseType> {
    public:
        iterator_type(PyArrayObject* array)
            :iterator_base<BaseType>(array) {
            }
        BaseType operator * () const {
            BaseType res;
            std::memcpy(&res,this->data_,sizeof(res));
            return res;
        }
};

template <typename BaseType>
class aligned_iterator_type : public iterator_base<BaseType> {
    public:
        aligned_iterator_type(PyArrayObject* array)
            :iterator_base<BaseType>(array) {
                assert(PyArray_ISALIGNED(array));
            }
        BaseType& operator * () const {
            return *this->data_;
        }
};


template <typename BaseType>
class array_base {
    protected:
        PyArrayObject* array_;

        void* raw_data(const position& pos) const {
            assert(this->validposition(pos));
            return PyArray_GetPtr(array_,const_cast<npy_intp*>(pos.position_));
        }
    public:
        array_base(const array_base<BaseType>& other)
            :array_(other.array_)
            {
                if (sizeof(BaseType) != PyArray_ITEMSIZE(array_)) {
                    std::cerr << "mahotas:" << __PRETTY_FUNCTION__ << " mix up of array types"
                        << " [using size " <<sizeof(BaseType) << " expecting " << PyArray_ITEMSIZE(array_) << "]\n";
                    assert(false);
                }
                Py_INCREF(array_);
            }

        array_base(PyArrayObject* array)
            :array_(array)
            {
                if (sizeof(BaseType) != PyArray_ITEMSIZE(array_)) {
                    std::cerr << "mahotas:" << __PRETTY_FUNCTION__ << " mix up of array types"
                        << " [using size " <<sizeof(BaseType) << " expecting " << PyArray_ITEMSIZE(array_) << "]\n";
                    assert(false);
                }
                Py_INCREF(array_);
            }

        ~array_base() {
            Py_XDECREF(array_);
        }
        array_base<BaseType>& operator = (const BaseType& other) {
            array_base<BaseType> na(other);
            this->swap(na);
        }
        void swap(array_base<BaseType>& other) {
            std::swap(this->array_, other.array_);
        }

        index_type size() const { return PyArray_SIZE(array_); }
        index_type size(index_type i) const {
            return this->dim(i);
        }
        index_type ndims() const { return PyArray_NDIM(array_); }
        index_type dim(index_type i) const {
            assert(i < this->ndims());
            return PyArray_DIM(array_,i);
        }


        unsigned stride(unsigned i) const {
            return PyArray_STRIDE(array_, i)/sizeof(BaseType);
        }
        PyArrayObject* raw_array() const { return array_; }
        void* raw_data() const { return PyArray_DATA(array_); }
        const npy_intp* raw_dims() const { return array_->dimensions; }

        bool validposition(const position& pos) const {
            if (ndims() != pos.nd_) return false;
            for (int i=0; i != pos.nd_; ++i) {
                if (pos[i] < 0 || pos[i] >= this->dim(i)) return false;
            }
            return true;
        }
        bool is_aligned() const {
            return PyArray_ISALIGNED(array_);
        }

        BaseType at(const position& pos) const {
            BaseType val;
            void* datap=raw_data(pos);
            memcpy(&val,datap,sizeof(BaseType));
            return val;
        }
        npy_intp raw_stride(npy_intp i) const {
            return PyArray_STRIDE(this->array_, i);
        }

};

template<typename BaseType>
struct array : public array_base<BaseType> {
    public:
        array(PyArrayObject* array)
            :array_base<BaseType>(array) {
            }
        typedef iterator_type<BaseType> iterator;
        typedef iterator_type<const BaseType> const_iterator;

        iterator begin() {
            return iterator(this->array_);
        }
        iterator end() {
            iterator res = begin();
            for (unsigned i = 0, N = this->size(); i!= N; ++i) {
                ++res;
            }
            return res;
        }

};

template <typename BaseType>
struct aligned_array : public array_base<BaseType> {
    private:
        bool is_carray_;
    public:
        aligned_array(PyArrayObject* array)
            :array_base<BaseType>(array)
            ,is_carray_(PyArray_ISCARRAY(array))
            {
                assert(PyArray_ISALIGNED(array));
            }
        aligned_array(const aligned_array<BaseType>& other)
            :array_base<BaseType>(other)
            ,is_carray_(other.is_carray_)
            { }
        typedef aligned_iterator_type<BaseType> iterator;
        typedef aligned_iterator_type<const BaseType> const_iterator;

        const_iterator begin() const {
            return const_iterator(this->array_);
        }
        iterator begin() {
            return iterator(this->array_);
        }
        iterator end() {
            iterator res = begin();
            for (unsigned i = 0, N = this->size(); i!= N; ++i) {
                ++res;
            }
            return res;
        }

        npy_intp stride(npy_intp i) const {
            return this->raw_stride(i)/sizeof(BaseType);
        }

        BaseType* data() {
            return reinterpret_cast<BaseType*>(PyArray_DATA(this->array_));
        }

        BaseType* data(npy_intp p0) {
            assert(p0 < this->dim(0));
            return reinterpret_cast<BaseType*>(PyArray_GETPTR1(this->array_, p0));
        }

        BaseType* data(npy_intp p0, npy_intp p1) {
            assert(p0 < this->dim(0));
            assert(p1 < this->dim(1));
            return reinterpret_cast<BaseType*>(PyArray_GETPTR2(this->array_, p0, p1));
        }

        BaseType* data(npy_intp p0, npy_intp p1, npy_intp p2) {
            assert(p0 < this->dim(0));
            assert(p1 < this->dim(1));
            assert(p2 < this->dim(2));
            return reinterpret_cast<BaseType*>(PyArray_GETPTR3(this->array_, p0, p1, p2));
        }

        const BaseType* data() const {
            return reinterpret_cast<const BaseType*>(PyArray_DATA(this->array_));
        }
        const BaseType* data(const position& pos) const {
            return reinterpret_cast<const BaseType*>(this->raw_data(pos));
        }

        const BaseType* data(npy_intp p0) const {
            assert(p0 < this->dim(0));
            return reinterpret_cast<const BaseType*>(PyArray_GETPTR1(this->array_, p0));
        }

        const BaseType* data(npy_intp p0, npy_intp p1) const {
            assert(p0 < this->dim(0));
            assert(p1 < this->dim(1));
            return reinterpret_cast<const BaseType*>(PyArray_GETPTR2(this->array_, p0, p1));
        }

        const BaseType* data(npy_intp p0, npy_intp p1, npy_intp p2) const {
            assert(p0 < this->dim(0));
            assert(p1 < this->dim(1));
            assert(p2 < this->dim(2));
            return reinterpret_cast<const BaseType*>(PyArray_GETPTR3(this->array_, p0, p1, p2));
        }


        BaseType* data(const position& pos) {
            return reinterpret_cast<BaseType*>(this->raw_data(pos));
        }

        BaseType& at(const position& pos) {
            return *data(pos);
        }
        BaseType at(const position& pos) const {
            return *data(pos);
        }

        BaseType& at_flat(npy_intp p) {
            if (is_carray_) return data()[p];

            BaseType* base = this->data();
            for (int d = this->ndims() - 1; d >= 0; --d) {
                int c = (p % this->dim(d));
                p /= this->dim(d-1);
                base += c * this->stride(d);
            }
            return *base;
        }
        BaseType at_flat(npy_intp p) const {
            return const_cast< aligned_array<BaseType>* >(this)->at_flat(p);
        }

        int pos_to_flat(const position& pos) const {
            npy_intp res = 0;
            int cummul = 1;
            for (int d = this->ndims() -1; d >= 0; --d) {
                res += pos.position_[d] * cummul;
                cummul *= this->dim(d);
            }
            return res;
        }
        numpy::position flat_to_pos(int p) const {
            numpy::position res;
            res.nd_ = this->ndims();
            for (int d = this->ndims() - 1; d >= 0; --d) {
                 res.position_[d] = (p % this->dim(d));
                 p /= this->dim(d);
            }
            if (p) res.position_[0] += p * this->dim(0);
            return res;
        }
        BaseType at(int p0) const {
            return *static_cast<BaseType*>(PyArray_GETPTR1(this->array_, p0));
        }
        BaseType& at(int p0) {
            assert(p0 < this->dim(0));
            return *static_cast<BaseType*>(PyArray_GETPTR1(this->array_, p0));
        }
        BaseType at(int p0, int p1) const {
            assert(p0 < this->dim(0));
            assert(p1 < this->dim(1));
            return *static_cast<BaseType*>(PyArray_GETPTR2(this->array_, p0, p1));
        }
        BaseType& at(int p0, int p1) {
            assert(p0 < this->dim(0));
            assert(p1 < this->dim(1));
            return *static_cast<BaseType*>(PyArray_GETPTR2(this->array_, p0, p1));
        }
        BaseType at(int p0, int p1, int p2) const {
            assert(p0 < this->dim(0));
            assert(p1 < this->dim(1));
            assert(p2 < this->dim(2));
            return *static_cast<BaseType*>(PyArray_GETPTR3(this->array_, p0, p1, p2));
        }
        BaseType& at(int p0, int p1, int p2) {
            assert(p0 < this->dim(0));
            assert(p1 < this->dim(1));
            assert(p2 < this->dim(2));
            return *static_cast<BaseType*>(PyArray_GETPTR3(this->array_, p0, p1, p2));
        }
};

template <typename BaseType>
aligned_array<BaseType> new_array(const npy_intp ndims, const npy_intp* dims) {
    aligned_array<BaseType> res(reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(ndims, const_cast<npy_intp*>(dims), dtype_code<BaseType>())));
    // SimpleNew returns an object with count = 1
    // constructing an array sets it to 2.
    Py_XDECREF(res.raw_array());
    return res;
}

template <typename BaseType>
aligned_array<BaseType> new_array(int s0) {
    npy_intp dim = s0;
    return new_array<BaseType>(1, &dim);
}
template <typename BaseType>
aligned_array<BaseType> new_array(int s0, int s1) {
    npy_intp dims[2];
    dims[0] = s0;
    dims[1] = s1;
    return new_array<BaseType>(2, dims);
}
template <typename BaseType>
aligned_array<BaseType> new_array(int s0, int s1, int s2) {
    npy_intp dims[3];
    dims[0] = s0;
    dims[1] = s1;
    dims[2] = s2;
    return new_array<BaseType>(3, dims);
}

template <typename BaseType>
aligned_array<BaseType> array_like(const array_base<BaseType>& orig) {
    PyArrayObject* array = orig.raw_array();
    return aligned_array<BaseType>((PyArrayObject*)PyArray_SimpleNew(array->nd,array->dimensions,PyArray_TYPE(array)));
}

} // namespace numpy

#endif // MAHOTAS_NUMPYPP_ARRAY_HPP_INCLUDE_GUARD_LPC_

