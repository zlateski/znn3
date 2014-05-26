#pragma once

#include "../core/types.hpp"
#include "../core/cube_pool.hpp"

#include "constant_convolve.hpp"

#include <zi/assert.hpp>

namespace zi {
namespace znn {

template<typename T>
inline void convolve_add(const cube<T>& a, const cube<T>& b, cube<T>& r)
{
    if ( size(b) == vec3s::one )
    {
        constant_convolve_add(a,b(0,0,0),r);
        return;
    }

    size_t ax = a.n_rows;
    size_t ay = a.n_cols;
    size_t az = a.n_slices;

    size_t bx = b.n_rows;
    size_t by = b.n_cols;
    size_t bz = b.n_slices;

    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    ZI_ASSERT(r.n_rows==rx);
    ZI_ASSERT(r.n_cols==ry);
    ZI_ASSERT(r.n_slices==rz);

    for ( size_t z = 0; z < rz; ++z )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t x = 0; x < rx; ++x )
            {
                for ( size_t dz = z, wz = bz - 1; dz < bz + z; ++dz, --wz )
                    for ( size_t dy = y, wy = by - 1; dy < by + y; ++dy, --wy )
                        for ( size_t dx = x, wx = bx - 1; dx < bx + x; ++dx, --wx )
                        {
                            r(x,y,z) += a(dx,dy,dz) * b(wx,wy,wz);
                        }
            }
}

template<typename T>
inline void convolve(const cube<T>& a, const cube<T>& b, cube<T>& r)
{
    if ( size(b) == vec3s::one )
    {
        constant_convolve(a,b(0,0,0),r);
        return;
    }

    r.fill(0);
    convolve_add(a,b,r);
}

template<typename T>
inline unique_cube<T> convolve(const cube<T>& a, const cube<T>& b)
{
    unique_cube<T> r = pool<T>::get_unique(vec3s::one + size(a) - size(b));
    convolve(a,b,*r);
    return r;
}


template<typename T>
inline void convolve_flipped_add(const cube<T>& a, const cube<T>& b, cube<T>& r)
{
    if ( size(a) == size(b) )
    {
        ZI_ASSERT(size(r) == vec3s::one);
        r(0,0,0) += constant_convolve_flipped(a,b);
        return;
    }

    size_t ax = a.n_rows;
    size_t ay = a.n_cols;
    size_t az = a.n_slices;

    size_t bx = b.n_rows;
    size_t by = b.n_cols;
    size_t bz = b.n_slices;

    size_t rx = ax - bx + 1;
    size_t ry = ay - by + 1;
    size_t rz = az - bz + 1;

    ZI_ASSERT(r.n_rows==rx);
    ZI_ASSERT(r.n_cols==ry);
    ZI_ASSERT(r.n_slices==rz);

    for ( size_t z = 0; z < rz; ++z )
        for ( size_t y = 0; y < ry; ++y )
            for ( size_t x = 0; x < rx; ++x )
            {
                for ( size_t dz = 0; dz < bz; ++dz )
                    for ( size_t dy = 0; dy < by; ++dy )
                        for ( size_t dx = 0; dx < bx; ++dx )
                        {
                            r(x,y,z) +=
                                a(ax-1-x-dx,ay-1-y-dy,az-1-z-dz) *
                                b(bx-1-dx,by-1-dy,bz-1-dz);
                        }
            }
}

template<typename T>
inline void convolve_flipped(const cube<T>& a, const cube<T>& b, cube<T>& r)
{
    if ( size(a) == size(b) )
    {
        ZI_ASSERT(size(r) == vec3s::one);
        r(0,0,0) = constant_convolve_flipped(a,b);
        return;
    }

    r.fill(0);
    convolve_flipped_add(a,b,r);
}

template<typename T>
inline unique_cube<T> convolve_flipped(const cube<T>& a, const cube<T>& b)
{
    unique_cube<T> r = pool<T>::get_unique(vec3s::one + size(a) - size(b));
    convolve_flipped(a,b,*r);
    return r;
}


template<typename T>
inline void convolve_inverse_add(const cube<T>& a, const cube<T>& b, cube<T>& r)
{
    if ( size(b) == vec3s::one )
    {
        constant_convolve_inverse_add(a,b(0,0,0),r);
        return;
    }

    size_t ax = a.n_rows;
    size_t ay = a.n_cols;
    size_t az = a.n_slices;

    size_t bx = b.n_rows;
    size_t by = b.n_cols;
    size_t bz = b.n_slices;

    size_t rx = ax + bx - 1;
    size_t ry = ay + by - 1;
    size_t rz = az + bz - 1;

    ZI_VERIFY(r.n_rows==rx);
    ZI_VERIFY(r.n_cols==ry);
    ZI_VERIFY(r.n_slices==rz);

    for ( size_t dy = 0; dy < by; ++dy )
        for ( size_t dz = 0; dz < bz; ++dz )
            for ( size_t dx = 0; dx < bx; ++dx )
            {
                size_t fx = bx - 1 - dx;
                size_t fy = by - 1 - dy;
                size_t fz = bz - 1 - dz;

                for ( size_t z = 0; z < az; ++z )
                    for ( size_t y = 0; y < ay; ++y )
                        for ( size_t x = 0; x < ax; ++x )
                        {
                            r(x+fx,y+fy,z+fz) += a(x,y,z) * b(dx,dy,dz);
                        }
            }
}

template<typename T>
inline void convolve_inverse(const cube<T>& a, const cube<T>& b, cube<T>& r)
{
    if ( size(b) == vec3s::one )
    {
        constant_convolve_inverse(a,b(0,0,0),r);
        return;
    }

    r.fill(0);
    convolve_inverse_add(a,b,r);
}

template<typename T>
inline unique_cube<T> convolve_inverse(const cube<T>& a, const cube<T>& b)
{
    unique_cube<T> r = pool<T>::get_unique(size(a) + size(b) - vec3s::one);
    convolve_inverse(a,b,*r);
    return r;
}


}} // namespace zi::znn
