#pragma once

#include "../core/types.hpp"
#include "../core/cube_utils.hpp"
#include "../core/cube_utils.hpp"

#include <functional>
#include <cstddef>
#include <utility>
#include <cstdint>

namespace zi {
namespace znn {

template<typename T, typename F>
inline std::pair<unique_cube<T>, unique_cube<uint32_t>>
    pooling_filter_2x2(const cube<T>& input_cube, F func, const vec3s& s)
{
    unique_cube<T> cp          = pool<T>::get_unique(size(input_cube));
    unique_cube<uint32_t> idxp = pool<uint32_t>::get_unique(size(input_cube));

    cube<T>& c                 = *cp;
    cube<uint32_t>& idx        = *idxp;

    c = input_cube;

    fill_indices(idx);

    std::pair<unique_cube<T>, unique_cube<uint32_t>> ret
        ( pool<T>::get_unique(size(input_cube) - vec3s::one),
          pool<uint32_t>::get_unique(size(input_cube) - vec3s::one) );

    // x direction
    for ( size_t z = 0; z < c.n_slices; ++z )
        for ( size_t y = 0; y < c.n_cols; ++y )
            for ( size_t x = 0; x < c.n_rows-1; ++x )
                if ( func(c(x+1,y,z),c(x,y,z)) )
                {
                    c(x,y,z) = c(x+1,y,z);
                    idx(x,y,z) = idx(x+1,y,z);
                }

    // y direction
    for ( size_t z = 0; z < c.n_slices; ++z )
        for ( size_t y = 0; y < c.n_cols-1; ++y )
            for ( size_t x = 0; x < c.n_rows-1; ++x )
                if ( func(c(x,y+1,z),c(x,y,z)) )
                {
                    c(x,y,z) = c(x,y+1,z);
                    idx(x,y,z) = idx(x,y+1,z);
                }

    cube<T>& rc        = *ret.first ;
    cube<uint32_t>& ri = *ret.second;

    for ( size_t z = 0; z < c.n_slices-1; ++z )
        for ( size_t y = 0; y < c.n_cols-1; ++y )
            for ( size_t x = 0; x < c.n_rows-1; ++x )
                if ( func(c(x,y,z+1),c(x,y,z)) )
                {
                    rc(x,y,z) = c(x,y,z+1);
                    ri(x,y,z) = idx(x,y,z+1);
                }
                else
                {
                    rc(x,y,z) = c(x,y,z);
                    ri(x,y,z) = idx(x,y,z);
                }

    return ret;
}

template<typename T>
inline unique_cube<T> pooling_filter_2x2_undo( const cube<T>& c,
                                               const cube<uint32_t>& idx )
{
    unique_cube<T> rp = pool<T>::get_unique_zero(size(c) + vec3s::one);

    T* rmem = rp->memptr();
    const T* cmem = c.memptr();
    const uint32_t* imem = idx.memptr();

    for ( size_t i = 0; i < c.n_elem; ++i )
    {
        rmem[imem[i]] = cmem[i];
    }

    return rp;
}

template<typename T>
inline unique_cube<T> pooling_filter_2x2_bprop( const cube<T>& c,
                                                const cube<uint32_t>& idx )
{
    unique_cube<T> rp = pool<T>::get_unique_zero(size(c) + vec3s::one);

    T* rmem = rp->memptr();
    const T* cmem = c.memptr();
    const uint32_t* imem = idx.memptr();

    for ( size_t i = 0; i < c.n_elem; ++i )
    {
        rmem[imem[i]] += cmem[i];
    }

    return rp;
}



}} // namespace zi::znn
