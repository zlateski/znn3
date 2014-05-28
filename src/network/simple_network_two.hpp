#pragma once

#include <vector>
#include <stdexcept>
#include <functional>

#include "layered_network.hpp"
#include "layered_network_data.hpp"
#include "../transfer_fn/transfer_fn.hpp"
#include "../core/cube_utils.hpp"
#include "../core/cube_pool.hpp"
#include "../convolution/convolve.hpp"
#include "../convolution/sparse_convolve.hpp"
#include "../pooling/pooling_filter_2.hpp"

namespace zi {
namespace znn {

class simple_network_two
{
private:
    typedef std::vector<cube<double>> cubes_type;

private:
    layered_network_data& net_;
    transfer_fn           transfer_fn_;

    std::vector<vec3s>                              sparsness_;
    std::vector<std::vector<unique_cube<uint32_t>>> pooling_indices_;

private:
    void forward_layer(std::size_t l, const vec3s& sparse)
    {
        ZI_ASSERT(l<net_.num_layers());

        sparsness_.push_back(sparse);

        network_layer& layer = net_.layer(l);

        std::size_t nin  = layer.num_inputs();
        std::size_t nout = layer.num_outputs();

        if ( net_.pooling_size(l) != vec3s::one )
        {
            pooling_indices_[l].resize(nout);
        }

        for ( std::size_t j = 0; j < nout; ++j )
        {
            for ( std::size_t i = 0; i < nin; ++i )
            {
                auto f = sparse_convolve(*net_.input_featuremap(l,i),
                                         layer.filter(i,j), sparse);
                if ( i == 0 )
                {
                    net_.featuremap(l,j) = std::move(f);
                }
                else
                {
                    *net_.featuremap(l,j) += *f;
                }
            }

            transfer_fn_.add_apply(layer.bias(j), *net_.featuremap(l,j));

            if ( net_.pooling_size(l) != vec3s::one )
            {
                auto pooled =
                    pooling_filter_2(*net_.featuremap(l,j),
                                     std::greater<double>(),
                                     net_.pooling_size(l),
                                     sparse);

                net_.featuremap(l,j)   = std::move(pooled.first);
                pooling_indices_[l][j] = std::move(pooled.second);
            }
        }
    }

    void backward_layer(std::size_t l, cubes_type& grads,
                        cubes_type& igrads)
    {
        ZI_ASSERT(l<net_.num_layers());
        ZI_ASSERT(grads.size()>0);
        ZI_ASSERT(grads.size()==net_.layer(l).num_outputs());

        network_layer& layer = net_.layer(l);

        std::size_t nin  = layer.num_inputs();
        std::size_t nout = layer.num_outputs();

        for ( std::size_t j = 0; j < nout; ++j )
        {
            transfer_fn_.apply_grad(grads[j], *net_.featuremap(l,j));

            net_.dEdB(l,j) = arma::accu(grads[j]);

            if ( net_.pooling_size(l) != vec3s::one )
            {
                auto g = pooling_filter_2_bprop(
                    grads[j],
                    *pooling_indices_[l][j],
                    net_.pooling_size(l),
                    sparsness_[l]);

                grads[j] = *g;
            }
        }

        if ( l > 0 )
        {
            igrads.resize(nin);
        }

        for ( std::size_t i = 0; i < nin; ++i )
        {
            for ( std::size_t j = 0; j < nout; ++j )
            {
                net_.dEdW(l,i,j) = sparse_convolve_flipped(
                    *net_.input_featuremap(l,i),
                    grads[j], sparsness_[l]);

                if ( l > 0 )
                {
                    auto g = sparse_convolve_inverse(grads[j],
                                                     net_.filter(l,i,j),
                                                     sparsness_[l]);
                    if ( j == 0 )
                    {
                        igrads[i] = *g;
                    }
                    else
                    {
                        igrads[i] += *g;
                    }
                }
            }
        }
    }

public:
    simple_network_two(layered_network_data& net, transfer_fn tf)
        : net_(net)
        , transfer_fn_(tf)
    {
        ZI_ASSERT(net.num_layers()>0);
    }

    cubes_type forward(const cubes_type& input)
    {
        sparsness_.clear();
        pooling_indices_.clear();
        pooling_indices_.resize(net_.num_layers());

        ZI_ASSERT(input.size()>0);
        ZI_ASSERT(input.size()==net_.num_inputs());

        for ( size_t i = 0; i < net_.num_inputs(); ++i )
        {
            net_.input(i) = pool<double>::get_unique_copy(input[i]);
        }

        vec3s sparse = vec3s::one;
        for ( std::size_t i = 0; i < net_.num_layers(); ++i )
        {
            forward_layer(i, sparse);
            sparse *= net_.pooling_size(i);
        }

        cubes_type ret;

        for ( size_t i = 0; i < net_.num_outputs(); ++i )
        {
            ret.push_back(*net_.output(i));
        }

        return ret;
    }

    void backward(const cubes_type& grads)
    {
        ZI_ASSERT(grads.size()>0);
        ZI_ASSERT(grads.size()==net_.num_outputs());

        cubes_type ograds = grads;
        cubes_type igrads;
        backward_layer(net_.num_layers()-1, ograds, igrads);

        for ( std::size_t l = net_.num_layers() - 1; l > 0; --l )
        {
            cubes_type ograds;
            std::swap(ograds, igrads);
            backward_layer(l-1, ograds, igrads);
        }
    }

    void grad_update()
    {
        net_.apply_grads_serial();
    }


}; // class simple_network_two

}} // namespace zi::znn
