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
        size_t z = c.n_slices / 2 + 1;
        for ( size_t y = 0; y < c.n_cols/2; ++y )
            for ( size_t x = 0; x < c.n_rows; ++x )
                std::swap(c(x,y,z),
                          c(c.n_rows-1-x,c.n_cols-1-y,c.n_slices-1-z));

        if ( c.n_cols % 2 )
        {
            size_t y = c.n_cols / 2 + 1;
            for ( size_t x = 0; x < c.n_rows; ++x )
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
    unique_cube<T> ret = pool<T>::get(s);
    *ret = c.subcube(0,0,0,s[0]-1,s[1]-1,s[2]-1);
    return ret;
}

}} // namespace zi::znn
