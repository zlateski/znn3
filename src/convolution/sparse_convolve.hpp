#pragma once

#include "../core/types.hpp"
#include "../core/cube_pool.hpp"
#include "convolve.hpp"

#include <zi/assert.hpp>

namespace zi {
namespace znn {

template<typename T>
inline void sparse_convolve_add(const cube<T>& a, const cube<T>& b,
                                const vec3s& s, cube<T>& r)
{
    if ( s == vec3s::one )
    {
        convolve_add(a, b, r);
        return;
    }

    size_t ax = a.n_rows;
    size_t ay = a.n_cols;
    size_t az = a.n_slices;

    size_t bx = b.n_rows;
    size_t by = b.n_cols;
    size_t bz = b.n_slices;

    size_t rbx = (bx-1) * s[0] + 1;
    size_t rby = (by-1) * s[1] + 1;
    size_t rbz = (bz-1) * s[2] + 1;

    size_t rx = ax - rbx + 1;
    size_t ry = ay - rby + 1;
    size_t rz = az - rbz + 1;

    ZI_ASSERT(r.n_rows==rx);
    ZI_ASSERT(r.n_cols==ry);
    ZI_ASSERT(r.n_slices==rz);

    for ( size_t z = 0; z < rz; ++z )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t x = 0; x < rx; ++x )
            {
                for ( size_t dz = z, wz = bz - 1; dz < rbz + z; dz += s[2], --wz )
                    for ( size_t dy = y, wy = by - 1; dy < rby + y; dy += s[1], --wy )
                        for ( size_t dx = x, wx = bx - 1; dx < rbx + x; dx += s[0], --wx )
                        {
                            r(x,y,z) += a(dx,dy,dz) * b(wx,wy,wz);
                        }
            }
}

template<typename T>
inline void sparse_convolve(const cube<T>& a, const cube<T>& b,
                            const vec3s& s, cube<T>& r)
{
    if ( s == vec3s::one )
    {
        convolve(a, b, r);
    }
    else
    {
        r.fill(0);
        sparse_convolve_add(a,b,s,r);
    }
}

template<typename T>
inline unique_cube<T> sparse_convolve(const cube<T>& a, const cube<T>& b,
                                      const vec3s& s)
{
    if ( s == vec3s::one )
    {
        return convolve(a,b);
    }

    vec3s ws = size(b);
    ws = (ws - vec3s::one) * s + vec3s::one;

    unique_cube<T> r = pool<T>::get_unique(vec3s::one + size(a) - ws);
    sparse_convolve(a,b,s,*r);
    return r;
}


template<typename T>
inline void sparse_convolve_flipped_add(const cube<T>& a, const cube<T>& b,
                                        const vec3s& s, cube<T>& r)
{
    if ( s == vec3s::one )
    {
        convolve_flipped_add(a, b, r);
        return;
    }

    size_t ax = a.n_rows;
    size_t ay = a.n_cols;
    size_t az = a.n_slices;

    size_t bx = b.n_rows;
    size_t by = b.n_cols;
    size_t bz = b.n_slices;

    size_t rbx = (bx-1) * s[0] + 1;
    size_t rby = (by-1) * s[1] + 1;
    size_t rbz = (bz-1) * s[2] + 1;

    size_t rx = ax - rbx + 1;
    size_t ry = ay - rby + 1;
    size_t rz = az - rbz + 1;

    ZI_ASSERT(r.n_rows==rx);
    ZI_ASSERT(r.n_cols==ry);
    ZI_ASSERT(r.n_slices==rz);

    for ( size_t z = 0; z < rz; ++z )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t x = 0; x < rx; ++x )
            {
                for ( size_t dz = 0, wz = bz - 1; dz < rbz; dz += s[2], --wz )
                    for ( size_t dy = 0, wy = by - 1; dy < rby; dy += s[1], --wy )
                        for ( size_t dx = 0, wx = bx - 1; dx < rbx; dx += s[0], --wx )
                        {
                            r(x,y,z) +=
                                a(ax-1-x-dx,ay-1-y-dy,az-1-z-dz) * b(wx,wy,wz);
                        }
            }
}

template<typename T>
inline void sparse_convolve_flipped(const cube<T>& a, const cube<T>& b,
                                    const vec3s& s, cube<T>& r)
{
    if ( s == vec3s::one )
    {
        convolve_flipped(a, b, r);
    }
    else
    {
        r.fill(0);
        sparse_convolve_flipped_add(a,b,s,r);
    }
}

template<typename T>
inline unique_cube<T> sparse_convolve_flipped(const cube<T>& a,
                                              const cube<T>& b,
                                              const vec3s& s)
{
    if ( s == vec3s::one )
    {
        return convolve_flipped(a,b);
    }

    vec3s ws = size(b);
    ws = (ws - vec3s::one) * s + vec3s::one;

    unique_cube<T> r = pool<T>::get_unique(vec3s::one + size(a) - ws);

    sparse_convolve_flipped(a,b,s,*r);
    return r;
}


template<typename T>
inline void sparse_convolve_inverse_add(const cube<T>& a, const cube<T>& b,
                                        const vec3s& s, cube<T>& r)
{
    if ( s == vec3s::one )
    {
        convolve_add(a, b, r);
        return;
    }

    size_t ax = a.n_rows;
    size_t ay = a.n_cols;
    size_t az = a.n_slices;

    size_t bx = b.n_rows;
    size_t by = b.n_cols;
    size_t bz = b.n_slices;

    size_t rbx = (bx-1) * s[0] + 1;
    size_t rby = (by-1) * s[1] + 1;
    size_t rbz = (bz-1) * s[2] + 1;

    size_t rx = ax + rbx - 1;
    size_t ry = ay + rby - 1;
    size_t rz = az + rbz - 1;

    ZI_VERIFY(r.n_rows==rx);
    ZI_VERIFY(r.n_cols==ry);
    ZI_VERIFY(r.n_slices==rz);

    for ( size_t wz = 0; wz < bz; ++wz )
        for ( size_t wy = 0; wy < by; ++wy )
            for ( size_t wx = 0; wx < bx; ++wx )
            {
                size_t fx = bx - 1 - wx;
                size_t fy = by - 1 - wy;
                size_t fz = bz - 1 - wz;

                size_t ox = fx * s[0];
                size_t oy = fy * s[1];
                size_t oz = fz * s[2];

                for ( size_t y = 0; y < ay; ++y )
                    for ( size_t z = 0; z < az; ++z )
                        for ( size_t x = 0; x < ax; ++x )
                        {
                            r(x+ox,y+oy,z+oz) += a(x,y,z) * b(wx,wy,wz);
                        }
            }
}

template<typename T>
inline void sparse_convolve_inverse(const cube<T>& a, const cube<T>& b,
                                    const vec3s& s, cube<T>& r)
{
    if ( s == vec3s::one )
    {
        convolve_inverse(a,b,r);
    }
    else
    {
        r.fill(0);
        sparse_convolve_inverse_add(a,b,s,r);
    }
}

template<typename T>
inline unique_cube<T> sparse_convolve_inverse(const cube<T>& a,
                                              const cube<T>& b,
                                              const vec3s& s)
{
    if ( s == vec3s::one )
    {
        return convolve_inverse(a,b);
    }

    vec3s ws = size(b);
    ws = (ws - vec3s::one) * s + vec3s::one;

    unique_cube<T> r = pool<T>::get_unique(size(a) + ws - vec3s::one);

    sparse_convolve_inverse(a,b,s,*r);
    return r;
}


}} // namespace zi::znn
