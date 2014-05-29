#pragma once

#include <cstddef>
#include <algorithm>
#include "../core/types.hpp"
#include "../core/cube_pool.hpp"

namespace zi {
namespace znn {
namespace frontiers {

inline std::tuple<size_t, double, double, cube<double>>
    square_loss( const sample& s, const cube<double> prop )
{
    cube<double> grad = prop;

    size_t n_samples = 0;
    double sq_err = 0;
    double cl_err = 0;

    for ( size_t z = 0; z < grad.n_slices; ++z )
        for ( size_t y = 0; y < grad.n_cols; ++y )
            for ( size_t x = 0; x < grad.n_rows; ++x )
            {
                if ( s.mask(x,y,z) )
                {
                    grad(x,y,z) = prop(x,y,z) - s.label(x,y,z);
                    grad(x,y,z) *= 2;
                    ++n_samples;

                    if ( s.label(x,y,z) )
                    {
                        grad(x,y,z) *= s.w_pos;
                        double e = prop(x,y,z) - s.label(x,y,z);
                        sq_err += e*e*s.w_pos*s.w_pos;;
                        if ( prop(x,y,z) < 0.5 )
                            cl_err += s.w_pos;
                    }
                    else
                    {
                        grad(x,y,z) *= s.w_neg;
                        double e = prop(x,y,z) - s.label(x,y,z);
                        sq_err += e*e*s.w_neg*s.w_neg;
                        if ( prop(x,y,z) >= 0.5 )
                            cl_err += s.w_neg;
                    }
                }
                else
                {
                    grad(x,y,z) = 0;
                }
            }

    return std::tuple<size_t, double, double, cube<double>>
    { n_samples, sq_err, cl_err, std::move(grad) };
}

}}} // namespace zi::znn::frontiers

