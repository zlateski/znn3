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
    cube<double> posw = prop;
    cube<double> negw = prop;

    size_t n_samples = 0;

    double tot_p_err = 0;
    double tot_n_err = 0;
    double cls_p_err = 0;
    double cls_n_err = 0;

    size_t w_pos = 0;
    size_t w_neg = 0;

    for ( size_t z = 0; z < grad.n_slices; ++z )
        for ( size_t y = 0; y < grad.n_cols; ++y )
            for ( size_t x = 0; x < grad.n_rows; ++x )
            {
                posw(x,y,z) = negw(x,y,z) = 0;

                if ( s.mask(x,y,z) )
                {
                    grad(x,y,z) = prop(x,y,z) - s.label(x,y,z);
                    grad(x,y,z) *= 2;
                    ++n_samples;

                    if ( s.label(x,y,z) )
                    {
                        posw(x,y,z) = 1;
                        ++w_pos;
                        double e = prop(x,y,z) - s.label(x,y,z);
                        tot_p_err += e*e;
                        if ( prop(x,y,z) < 0.5 )
                            ++cls_p_err;
                    }
                    else
                    {
                        negw(x,y,z) = 1;
                        ++w_neg;
                        double e = prop(x,y,z) - s.label(x,y,z);
                        tot_n_err += e*e;
                        if ( prop(x,y,z) >= 0.5 )
                            ++cls_n_err;
                    }
                }
                else
                {
                    grad(x,y,z) = 0;
                }
            }

    if ( w_pos )
    {
        cls_p_err /= w_pos;
        tot_p_err /= w_pos;
        posw /= w_pos;
    }

    if ( w_neg )
    {
        cls_n_err /= w_neg;
        tot_n_err /= w_neg;
        negw /= w_neg;
    }

    pairwise_mult(posw, grad);
    pairwise_mult(negw, grad);

    grad = posw + negw;
    grad *= static_cast<double>(n_samples)/2;

    return std::tuple<size_t, double, double, cube<double>>
    { n_samples,
            (tot_p_err + tot_n_err) * n_samples / 2,
            (cls_p_err + cls_n_err) * n_samples / 2,
            std::move(grad) };
}

}}} // namespace zi::znn::frontiers

