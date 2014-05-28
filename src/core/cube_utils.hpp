#pragma once

#include <cstddef>
#include <algorithm>
#include "types.hpp"
#include "cube_pool.hpp"

namespace zi {
namespace znn {

template<typename T>
inline bool equal(const cube<T>& c1, const cube<T>& c2)
{
    if ( c1.n_elem != c2.n_elem )
    {
        return false;
    }

    return std::equal(c1.memptr(), c1.memptr() + c1.n_elem, c2.memptr());
}

template<typename T>
inline cube<T> make_cube(const vec3s& s)
{
    return cube<T>(s[0],s[1],s[2]);
}

template<typename T>
inline cube<T> make_zero_cube(const vec3s& s)
{
    return arma::zeros<cube<T>>(s[0],s[1],s[2]);
}

template<typename T>
inline vec3s size(const cube<T>& c)
{
    return vec3s(c.n_rows, c.n_cols, c.n_slices);
}

template<typename T>
inline void flip_dims(cube<T>& c)
{
    for ( size_t z = 0; z < c.n_slices/2; ++z )
        for ( size_t y = 0; y < c.n_cols; ++y )
            for ( size_t x = 0; x < c.n_rows; ++x )
                std::swap(c(x,y,z),
                          c(c.n_rows-1-x,c.n_cols-1-y,c.n_slices-1-z));

    if ( c.n_slices % 2 )
    {
        size_t z = c.n_slices / 2;
        for ( size_t y = 0; y < c.n_cols/2; ++y )
            for ( size_t x = 0; x < c.n_rows; ++x )
                std::swap(c(x,y,z),
                          c(c.n_rows-1-x,c.n_cols-1-y,c.n_slices-1-z));

        if ( c.n_cols % 2 )
        {
            size_t y = c.n_cols / 2;
            for ( size_t x = 0; x < c.n_rows/2; ++x )
                std::swap(c(x,y,z),
                          c(c.n_rows-1-x,c.n_cols-1-y,c.n_slices-1-z));
        }
    }

}

template<typename T>
inline void fill_indices(cube<T>& c)
{
    T  idx = 0;
    T* mem = c.memptr();

    for ( size_t i = 0; i < c.n_elem; ++i, ++idx )
        mem[i] = idx;
}

template<typename T>
inline unique_cube<T> crop(cube<T>& c, const vec3s& s)
{
    unique_cube<T> ret = pool<T>::get_unique(s);
    *ret = c.subcube(0,0,0,s[0]-1,s[1]-1,s[2]-1);
    return ret;
}

template<typename T>
inline void sparse_explode(const cube<T>& in, cube<T>& out,
                           const vec3s& s)
{
    for ( size_t z = 0, zout = 0; z < in.n_slices; ++z, zout += s[2] )
        for ( size_t y = 0, yout = 0; y < in.n_cols; ++y, yout += s[1] )
            for ( size_t x = 0, xout = 0; x < in.n_rows; ++x, xout += s[0] )
                out(xout,yout,zout) = in(x,y,z);
}

template<typename T>
inline unique_cube<T> sparse_implode_flip(const cube<T>& in,
                                          const vec3s& sz,
                                          const vec3s& sp)
{
    unique_cube<T> r = pool<T>::get_unique(sz);
    for ( size_t z = 0, zin = in.n_slices-1; z < sz[2]; ++z, zin -= sp[2] )
        for ( size_t y = 0, yin = in.n_cols-1; y < sz[1]; ++y, yin -= sp[1] )
            for ( size_t x = 0, xin = in.n_rows-1; x < sz[0]; ++x, xin -= sp[0] )
                (*r)(x,y,z) = in(xin, yin, zin);
    return r;
}

template<typename T>
inline unique_cube<T> crop_right( const cube<T>& in, const vec3s& s )
{
    auto r = pool<T>::get_unique(s);
    vec3s b = size(in) - s;
    *r = in.subcube(b[0], b[1], b[2], b[0]+s[0]-1, b[1]+s[1]-1, b[2]+s[2]-1);
    return r;
}

template<typename T>
inline unique_cube<T> expand( const cube<T>& in, const vec3s& s )
{
    auto r = pool<T>::get_unique_zero(s);
    vec3s b = size(in);
    r->subcube(0,0,0,b[0]-1,b[1]-1,b[2]-1) = in;
    return r;
}

template<typename T>
inline void pairwise_mult( cube<T>& a, const cube<T>& b )
{
    ZI_ASSERT(a.n_elem==b.n_elem);
    T* ap = a.memptr();
    const T* bp = b.memptr();

    for ( size_t i = 0; i < a.n_elem; ++i )
        ap[i] *= bp[i];
}

}} // namespace zi::znn
