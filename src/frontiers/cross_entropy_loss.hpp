#pragma once

#include <cstddef>
#include <algorithm>
#include "../core/types.hpp"
#include "../core/cube_pool.hpp"
#include "../core/cube_utils.hpp"

namespace zi {
namespace znn {
namespace frontiers {

inline std::tuple<size_t, double, double, cube<double>, cube<double>>
    cross_entropy_loss( const sample& s,
                        const cube<double> a,
                        const cube<double> b )
{
    cube<double> ap = exp(a);
    cube<double> bp = exp(b);

    cube<double> sum = ap + bp;

    pairwise_div(ap,sum);
    pairwise_div(bp,sum);

    cube<double> agrad = ap;
    cube<double> bgrad = ap;

    size_t n_samples = 0;
    double lg_err = 0;
    double cl_err = 0;

    for ( size_t z = 0; z < grad.n_slices; ++z )
        for ( size_t y = 0; y < grad.n_cols; ++y )
            for ( size_t x = 0; x < grad.n_rows; ++x )
            {
                if ( s.mask(x,y,z) )
                {
                    ++n_samples;

                    agrad(x,y,z) = a(x,y,z) - s.label(x,y,z);
                    bgrad(x,y,z) = b(x,y,z) - 1 + s.label(x,y,z);

                    lg_err -= s.label(x,y,z) * std::log(a(x,y,z));
                    lg_err -= (static_cast<double>(1) - s.label(x,y,z))
                        * std::log(b(x,y,z));

                    if ( s.label(x,y,z) )
                    {
                        grad(x,y,z) *= s.w_pos;
                        double e = prop(x,y,z) - s.label(x,y,z);
                        sq_err += e*e*s.w_pos;
                        if ( prop(x,y,z) < 0.5 )
                            cl_err += s.w_pos;
                    }
                    else
                    {
                        grad(x,y,z) *= s.w_neg;
                        double e = prop(x,y,z) - s.label(x,y,z);
                        sq_err += e*e*s.w_neg;
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

