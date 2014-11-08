// Copyright: 2014 Luis Pedro Coelho
// License: MIT
#ifndef LPC_MAHOTAS_DEBUG_20140210_INCLUDE_GUARD_
#define LPC_MAHOTAS_DEBUG_20140210_INCLUDE_GUARD_


template <typename T>
struct checked_pointer {
    T* base_;
    int n_;
    checked_pointer(T* base, int n)
        :base_(base)
        ,n_(n)
    { }

    T& operator [] (unsigned ix) {
        assert(ix < unsigned(n_));
        return base_[ix];
    }

    T& operator *() { return (*this)[0]; }

    checked_pointer<T>& operator ++ () {
        ++base_;
        --n_;
        return *this;
    }
    checked_pointer<T> operator ++ (int) {
        checked_pointer<T> before = *this;
        ++(*this);
        return before;
    }
};

#ifdef _GLIBCXX_DEBUG

template<typename T>
struct get_pointer_type {
    typedef checked_pointer<T> ptr;
    typedef const checked_pointer<const T> cptr;
};

template <typename T>
checked_pointer<T> as_checked_ptr(T* p, const int n) {
    return checked_pointer<T>(p, n);
}

#else

template<typename T>
struct get_pointer_type {
    typedef T* ptr;
    typedef const T* cptr;
};

template <typename T>
T* as_checked_ptr(T* p, const int) {
    return p;
}

#endif

#endif // LPC_MAHOTAS_DEBUG_20140210_INCLUDE_GUARD_
