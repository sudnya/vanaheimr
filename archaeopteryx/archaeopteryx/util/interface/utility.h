/*	\file   utility.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 14, 2012
	\brief  The header file for stl utility classes.
*/

#pragma once

namespace archaeopteryx
{

namespace util
{


namespace rel_ops
{

template<class _Tp>
__device__ inline bool
operator!=(const _Tp& __x, const _Tp& __y)
{
    return !(__x == __y);
}

template<class _Tp>
__device__ inline bool
operator> (const _Tp& __x, const _Tp& __y)
{
    return __y < __x;
}

template<class _Tp>
__device__ inline bool
operator<=(const _Tp& __x, const _Tp& __y)
{
    return !(__y < __x);
}

template<class _Tp>
__device__ inline bool
operator>=(const _Tp& __x, const _Tp& __y)
{
    return !(__x < __y);
}

}  // rel_ops

// swap_ranges

template <class _ForwardIterator1, class _ForwardIterator2>
__device__ inline _ForwardIterator2
swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2)
{
    for(; __first1 != __last1; ++__first1, ++__first2)
        swap(*__first1, *__first2);
    return __first2;
}

template<class _Tp, size_t _Np>
__device__ inline void
swap(_Tp (&__a)[_Np], _Tp (&__b)[_Np])
{
    util::swap_ranges(__a, __a + _Np, __b);
}

template <class _Tp>
__device__ inline const _Tp&
move_if_noexcept(_Tp& __x)
{
    return util::move(__x);
}

struct piecewise_construct_t { };
extern const piecewise_construct_t piecewise_construct;

template <class _T1, class _T2>
struct pair
{
    typedef _T1 first_type;
    typedef _T2 second_type;

    _T1 first;
    _T2 second;

    // pair(const pair&) = default;
    // pair(pair&&) = default;

    __device__ pair() : first(), second() {}

    __device__ pair(const _T1& __x, const _T2& __y)
        : first(__x), second(__y) {}
		
    template<class _U1, class _U2>
                __device__ pair(const pair<_U1, _U2>& __p)
            : first(__p.first), second(__p.second) {}

    __device__ pair(const pair& __p)
        : first(__p.first),
          second(__p.second)
    {
    }

    __device__ pair& operator=(const pair& __p)
    {
        first = __p.first;
        second = __p.second;
        return *this;
    }
    
    __device__ void
    swap(pair& __p)
    {
        util::iter_swap(&first, &__p.first);
        util::iter_swap(&second, &__p.second);
    }
private:
};

template <class _T1, class _T2>
__device__ inline bool
operator==(const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return __x.first == __y.first && __x.second == __y.second;
}

template <class _T1, class _T2>
__device__ inline bool
operator!=(const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return !(__x == __y);
}

template <class _T1, class _T2>
__device__ inline bool
operator< (const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return __x.first < __y.first || (!(__y.first < __x.first) && __x.second < __y.second);
}

template <class _T1, class _T2>
__device__ inline bool
operator> (const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return __y < __x;
}

template <class _T1, class _T2>
__device__ inline bool
operator>=(const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return !(__x < __y);
}

template <class _T1, class _T2>
__device__ inline bool
operator<=(const pair<_T1,_T2>& __x, const pair<_T1,_T2>& __y)
{
    return !(__y < __x);
}

template <class _T1, class _T2>
__device__ inline void
swap(pair<_T1, _T2>& __x, pair<_T1, _T2>& __y)
{
    __x.swap(__y);
}

template <class _T1, class _T2>
__device__ inline pair<_T1,_T2>
make_pair(_T1 __x, _T2 __y)
{
    return pair<_T1, _T2>(__x, __y);
}

template <size_t _Ip> struct __get_pair;

template <>
struct __get_pair<0>
{
    template <class _T1, class _T2>
    __device__ static
        _T1&
    get(pair<_T1, _T2>& __p) {return __p.first;}

    template <class _T1, class _T2>
    __device__ static
        const _T1&
    get(const pair<_T1, _T2>& __p) {return __p.first;}

};

template <>
struct __get_pair<1>
{
    template <class _T1, class _T2>
    __device__ static
        _T2&
    get(pair<_T1, _T2>& __p) {return __p.second;}

    template <class _T1, class _T2>
    __device__ static
        const _T2&
    get(const pair<_T1, _T2>& __p) {return __p.second;}

};

}

}


