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

    double tot_p_err = 0;
    double tot_n_err = 0;
    double cls_p_err = 0;
    double cls_n_err = 0;

    size_t w_pos = 0;
    size_t w_neg = 0;

    for ( size_t z = 0; z < a.n_slices; ++z )
        for ( size_t y = 0; y < a.n_cols; ++y )
            for ( size_t x = 0; x < a.n_rows; ++x )
            {
                agrad(x,y,z) = bgrad(x,y,z) = 0;

                if ( s.mask(x,y,z) )
                {
                    ++n_samples;

                    if ( s.label(x,y,z) > 0.5 )
                    {
                        ++w_pos;
                        agrad(x,y,z) = ap(x,y,z) - 1;

                        tot_p_err -= std::log(ap(x,y,z));

                        if ( ap(x,y,z) < bp(x,y,z) )
                            ++cls_p_err;
                    }
                    else
                    {
                        ++w_neg;
                        bgrad(x,y,z) = bp(x,y,z) - 1;

                        tot_n_err -= std::log(bp(x,y,z));

                        if ( ap(x,y,z) >= bp(x,y,z) )
                            ++cls_n_err;
                    }
                }
            }

    if ( w_pos )
    {
        agrad *= static_cast<double>(n_samples)/2;
        agrad /= w_pos;
        tot_p_err /= w_pos;
        cls_p_err /= w_pos;
    }

    if ( w_neg )
    {
        bgrad *= static_cast<double>(n_samples)/2;
        bgrad /= w_neg;
        tot_n_err /= w_neg;
        cls_n_err /= w_neg;
    }

    return std::tuple<size_t, double, double, cube<double>, cube<double>>
    { n_samples,
            (tot_p_err + tot_n_err) * n_samples / 2,
            (cls_p_err + cls_n_err) * n_samples / 2,
            std::move(agrad), std::move(bgrad) };
}

}}} // namespace zi::znn::frontiers

