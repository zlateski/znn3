#pragma once

#include "../core/types.hpp"
#include "../core/cube_pool.hpp"

#include <zi/assert.hpp>

namespace zi {
namespace znn {

template<typename T>
inline void constant_convolve_add(const cube<T>& a,
                                  typename identity<T>::type b,
                                  cube<T>& r)
{
    ZI_ASSERT(size(a)==size(r));

    const T* ap = a.memptr();
    T* rp = r.memptr();

    for ( size_t i = 0; i < a.n_elem; ++i )
        rp[i] += ap[i] * b;
}

template<typename T>
inline void constant_convolve(const cube<T>& a,
                              typename identity<T>::type b,
                              cube<T>& r)
{
    ZI_ASSERT(size(a)==size(r));

    const T* ap = a.memptr();
    T* rp = r.memptr();

    for ( size_t i = 0; i < a.n_elem; ++i )
        rp[i] = ap[i] * b;
}

template<typename T>
inline unique_cube<T> constant_convolve(const cube<T>& a,
                                        typename identity<T>::type b)
{
    unique_cube<T> r = pool<T>::get_unique(size(a));
    constant_convolve(a,b,*r);
    return r;
}


template<typename T>
inline T constant_convolve_flipped(const cube<T>& a, const cube<T>& b)
{
    ZI_ASSERT(size(a)==size(b));

    T r = 0;

    const T* ap = a.memptr();
    const T* bp = b.memptr();

    for ( size_t i = 0; i < a.n_elem; ++i )
        r += ap[i] * bp[i];

    return r;
}

template<typename T>
inline void constant_convolve_inverse_add(const cube<T>& a,
                                          typename identity<T>::type b,
                                          cube<T>& r)
{
    constant_convolve_add(a,b,r);
}

template<typename T>
inline void constant_convolve_inverse(const cube<T>& a,
                                      typename identity<T>::type b,
                                      cube<T>& r)
{
    constant_convolve(a,b,r);
}

template<typename T>
inline unique_cube<T> constant_convolve_inverse(const cube<T>& a,
                                                typename identity<T>::type b)
{
    return constant_convolve(a,b);
}


}} // namespace zi::znn
