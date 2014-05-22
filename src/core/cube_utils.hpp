#pragma once

#include <cstddef>
#include <algorithm>
#include "types.hpp"

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

}} // namespace zi::znn
