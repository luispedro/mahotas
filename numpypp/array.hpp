/* Copyright 2008 (C)
 * Luís Pedro Coelho <lpc@cmu.edu>
 * License GPL Version 2.
 */

#include <iterator>
extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

#include <stdio.h>

namespace numpy_utils {

struct numpy_position_type {
    numpy_position_type()
        :nd(0) {
        }
    numpy_position_type(npy_intp* pos, int nd)
        :nd(nd)
        { for (int i = 0; i != nd; ++i) position[i]=pos[i]; }
    int nd;
    npy_intp position[NPY_MAXDIMS];
    bool operator == (const numpy_position_type& other) { return !std::memcmp(this->position,other.position,sizeof(int)*this->nd); }
    bool operator != (const numpy_position_type& other) { return !(*this == other); }
};
typedef numpy_position_type numpy_position;

numpy_position_type operator + (const numpy_position_type& a, const numpy_position_type& b) {
    assert(a.nd == b.nd);
    numpy_position_type res = a;
    for (int i = 0; i != a.nd; ++i) res.position[i] += b.position[i];
    return res;
}


template <typename BaseType>
struct numpy_iterator_type_base : std::iterator<std::forward_iterator_tag, BaseType>{
    protected:
        BaseType* data_;
        // steps is similar to strides, but more useful for iteration, see implementation of operator ++
        // Also, I divide by sizeof(BaseType)
        int steps[NPY_MAXDIMS];
        int dimensions[NPY_MAXDIMS];
        numpy_position_type position_;

    public:
        numpy_iterator_type_base(PyArrayObject* array) {
            position_.nd=array->nd;
            data_=reinterpret_cast<BaseType*>(array->data);
            for (int i = 0; i != position_.nd; ++i) position_.position[i]=0;
            unsigned cummul = 0;
            for (int i = 0; i != position_.nd; ++i) {
                dimensions[i] = array->dimensions[position_.nd-i-1];
                steps[i] = array->strides[position_.nd-i-1]/sizeof(BaseType)-cummul;
                cummul += array->strides[position_.nd-i-1]*array->dimensions[position_.nd-i-1]/sizeof(BaseType);
            }
        }

        numpy_iterator_type_base& operator ++ () {
            for (int i = 0; i != position_.nd; ++i) {
                data_ += steps[i];
                ++position_.position[i];
                if (position_.position[i] != dimensions[i]) {
                    return *this;
                }
                position_.position[i] = 0;
            }
            return *this;
        }

        bool operator == (const numpy_iterator_type_base& other) { return this->position_ == other.position_; }
        bool operator != (const numpy_iterator_type_base& other) { return !(*this == other); }

        numpy_position_type position() const { return position_; }
};

template <typename BaseType>
class numpy_iterator_type : public numpy_iterator_type_base<BaseType> {
    public:
        numpy_iterator_type(PyArrayObject* array)
            :numpy_iterator_type_base<BaseType>(array) {
            }
        BaseType operator * () {
            BaseType res;
            std::memcpy(&res,this->data_,sizeof(res));
            return res;
        }
};

template <typename BaseType>
class numpy_aligned_iterator_type : public numpy_iterator_type_base<BaseType> {
    public:
        numpy_aligned_iterator_type(PyArrayObject* array)
            :numpy_iterator_type_base<BaseType>(array) {
                assert(PyArray_ISALIGNED(array));
            }
        BaseType& operator * () {
            return *this->data_;
        }
};


template <typename BaseType>
class numpy_array_type_base {
    protected:
        PyArrayObject* array_;

    public:
        numpy_array_type_base(PyArrayObject* array)
            :array_(array)
            {
                Py_INCREF(array_);
            }

        ~numpy_array_type_base() {
            Py_XDECREF(array_);
        }
        
        unsigned size() const { return PyArray_SIZE(array_); }
        unsigned ndims() const { return PyArray_NDIM(array_); }
        unsigned dim(unsigned i) const {
            assert(i < this->ndims());
            return PyArray_DIM(array_,i);
        }

        bool validposition(const numpy_position_type& pos) {
            if (ndims() != pos.nd) return false;
            for (int i=0; i != pos.nd; ++i) {
                if (pos.position[i] < 0 || pos.position[i] >= this->dim(i)) return false;
            }
            return true;
        }
        bool is_aligned() const {
            return PyArray_ISALIGNED(array_);
        }

        BaseType at(const numpy_position_type& pos) const {
            assert(this->validposition(pos));
            BaseType val;
            void* datap=PyArray_GetPtr(array_,const_cast<npy_intp*>(pos.position));
            memcpy(&val,datap,sizeof(BaseType));
            return val;
        }
};

template<typename BaseType>
struct numpy_array : public numpy_array_type_base<BaseType> {
    public:
        numpy_array(PyArrayObject* array)
            :numpy_array_type_base<BaseType>(array) {
            }
        typedef numpy_iterator_type<BaseType> iterator;
        typedef numpy_iterator_type<const BaseType> const_iterator;

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
struct numpy_aligned_array : public numpy_array_type_base<BaseType> {
    public:
        numpy_aligned_array(PyArrayObject* array)
            :numpy_array_type_base<BaseType>(array) {
                assert(PyArray_ISALIGNED(array));
            }
        typedef numpy_aligned_iterator_type<BaseType> iterator;
        typedef numpy_aligned_iterator_type<const BaseType> const_iterator;

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

        BaseType* data() {
            return reinterpret_cast<BaseType*>PyArray_DATA(this->array_);
        }
        const BaseType* data(const numpy_position_type& pos) const {
            return reinterpret_cast<const BaseType*>(PyArray_GetPtr(this->array_,const_cast<npy_intp*>(pos.position)));
        }

        BaseType* data(const numpy_position_type& pos) {
            return reinterpret_cast<BaseType*>(PyArray_GetPtr(this->array_,const_cast<npy_intp*>(pos.position)));
        }

        BaseType& at(const numpy_position_type& pos) {
            return *data(pos);
        }
        BaseType at(const numpy_position_type& pos) const {
            return *data(pos);
        }
};

} // namespace numpy_utils

