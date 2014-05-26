#pragma once

#include "../core/types.hpp"
#include "../core/cube_utils.hpp"
#include "../core/tube_iterator.hpp"

#include <functional>
#include <cstddef>
#include <set>

namespace zi {
namespace znn {

template<typename It1, typename It2, typename F>
inline void pooling_filter_pass( It1 first1, It1 last1, It2 first2,
                                 size_t l, F cmp )
{
    typedef typename It1::value_type first_type;
    typedef typename It2::value_type second_type;

    typedef std::pair<first_type, second_type> pair_type;

    auto cmpf = [&cmp](const pair_type& l, const pair_type& r) {
        return cmp(l.first, r.first) ? true :
        ( cmp(r.first, l.first) ? false : ( l.second < r.second) );
    };

    std::set<pair_type, decltype(cmpf)> set(cmpf);

    It1 tail1 = first1;
    It2 tail2 = first2;

    ZI_ASSERT(l>0);

    for (; l > 1; --l )
    {
        set.emplace(*tail1++, *tail2++);
    }

    ZI_ASSERT(tail1<last1);

    while ( tail1 < last1 )
    {
        set.emplace(*tail1++, *tail2++);
        pair_type r = *set.begin();

        set.erase(pair_type(*first1, *first2));

        *first1++ = r.first;
        *first2++ = r.second;
    }
}

template<typename T, typename I, typename F>
inline void inplace_pooling_filter( cube<T>& c,
                                    cube<I>& idx,
                                    F f,
                                    const vec3s& ps,
                                    const vec3s& ss = vec3s::one)
{
    if ( ps[0] > 1 )
    {
        for ( size_t x = 0; x < ss[0]; ++x )
            for ( size_t y = 0; y < c.n_cols; ++y )
                for ( size_t z = 0; z < c.n_slices; ++z )
                    pooling_filter_pass( tube_begin(c,x,y,z,x_direction,ss[0]),
                                         tube_end(c,x,y,z,x_direction,ss[0]),
                                         tube_begin(idx,x,y,z,x_direction,ss[0]),
                                         ps[0], f );
    }

    if ( ps[1] > 1 )
    {
        for ( size_t x = 0; x < c.n_rows; ++x )
            for ( size_t y = 0; y < ss[1]; ++y )
                for ( size_t z = 0; z < c.n_slices; ++z )
                    pooling_filter_pass( tube_begin(c,x,y,z,y_direction,ss[1]),
                                         tube_end(c,x,y,z,y_direction,ss[1]),
                                         tube_begin(idx,x,y,z,y_direction,ss[1]),
                                         ps[1], f );
    }

    if ( ps[2] > 1 )
    {
        for ( size_t x = 0; x < c.n_rows; ++x )
            for ( size_t y = 0; y < c.n_cols; ++y )
                for ( size_t z = 0; z < ss[2]; ++z )
                    pooling_filter_pass( tube_begin(c,x,y,z,z_direction,ss[2]),
                                         tube_end(c,x,y,z,z_direction,ss[2]),
                                         tube_begin(idx,x,y,z,z_direction,ss[2]),
                                         ps[2], f );
    }

}
}} // namespace zi::znn
