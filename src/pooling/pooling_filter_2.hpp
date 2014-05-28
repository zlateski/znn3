#pragma once

#include "../core/types.hpp"
#include "../core/cube_utils.hpp"
#include "../core/cube_pool.hpp"

#include <functional>
#include <cstddef>
#include <utility>
#include <cstdint>

namespace zi {
namespace znn {

template<typename T, typename F>
inline std::pair<unique_cube<T>, unique_cube<uint32_t>>
    pooling_filter_2( const cube<T>& input_cube,
                      F compare,
                      const vec3s& fs,
                      const vec3s& ss)
{
    unique_cube<T> cube_pointer =
        pool<T>::get_unique_copy(input_cube);

    unique_cube<uint32_t> indices_pointer =
        pool<uint32_t>::get_unique(size(input_cube));

    cube<T>& cb             = *cube_pointer;
    cube<uint32_t>& indices = *indices_pointer;

    fill_indices(indices);

    ZI_ASSERT((fs[0]>0)&&(fs[0]<3)&&(fs[1]>0)&&
              (fs[1]<3)&&(fs[2]>0)&&(fs[2]<3));

    // x direction
    if ( fs[0] == 2 )
        for ( size_t z = 0; z < cb.n_slices; ++z )
            for ( size_t y = 0; y < cb.n_cols; ++y )
                for ( size_t x = 0; x < cb.n_rows-ss[0]; ++x )
                    if ( compare(cb(x+ss[0],y,z),cb(x,y,z)) )
                    {
                        cb     (x,y,z) = cb     (x+ss[0],y,z);
                        indices(x,y,z) = indices(x+ss[0],y,z);
                    }

    // y direction
    if ( fs[1] == 2 )
        for ( size_t z = 0; z < cb.n_slices; ++z )
            for ( size_t y = 0; y < cb.n_cols-ss[1]; ++y )
                for ( size_t x = 0; x < cb.n_rows-ss[0]; ++x )
                    if ( compare(cb(x,y+ss[1],z),cb(x,y,z)) )
                    {
                        cb     (x,y,z) = cb     (x,y+ss[1],z);
                        indices(x,y,z) = indices(x,y+ss[1],z);
                    }

    // z direction
    if ( fs[2] == 2 )
        for ( size_t z = 0; z < cb.n_slices-ss[2]; ++z )
            for ( size_t y = 0; y < cb.n_cols-ss[1]; ++y )
                for ( size_t x = 0; x < cb.n_rows-ss[0]; ++x )
                    if ( compare(cb(x,y,z+ss[2]),cb(x,y,z)) )
                    {
                        cb     (x,y,z) = cb     (x,y,z+ss[2]);
                        indices(x,y,z) = indices(x,y,z+ss[2]);
                    }

    vec3s ret_size = size(input_cube) - (fs-vec3s::one) * ss;

    return { pool<T>::get_unique_crop(cb,ret_size),
            pool<uint32_t>::get_unique_crop(indices,ret_size) };
}

template<typename T>
inline unique_cube<T> pooling_filter_2_undo( const cube<T>& c,
                                             const cube<uint32_t>& idx,
                                             const vec3s& fs,
                                             const vec3s& ss)

{
    vec3s ret_size = size(c) + (fs-vec3s::one) * ss;

    unique_cube<T> rp = pool<T>::get_unique_zero(ret_size);

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
inline unique_cube<T> pooling_filter_2_bprop( const cube<T>& c,
                                              const cube<uint32_t>& idx,
                                              const vec3s& fs,
                                              const vec3s& ss)

{
    vec3s ret_size = size(c) + (fs-vec3s::one) * ss;

    unique_cube<T> rp = pool<T>::get_unique_zero(ret_size);

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
