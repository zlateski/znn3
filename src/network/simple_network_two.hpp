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

namespace zi {
namespace znn {

class simple_network_two
{
private:
    typedef std::vector<cube<double>> cubes_type;

private:
    layered_network_data& net_;
    transfer_fn           transfer_fn_;

private:
    void forward_layer(std::size_t l)
    {
        ZI_ASSERT(l<net_.num_layers());

        network_layer& layer = net_.layer(l);

        std::size_t nin  = layer.num_inputs();
        std::size_t nout = layer.num_outputs();

        vec3s in_size  = size(*net_.input_featuremap(l,0));
        vec3s out_size = vec3s::one + in_size - layer.filter_size();

        for ( std::size_t j = 0; j < nout; ++j )
        {
            net_.featuremap(l, j) = pool<double>::get_unique_zero(out_size);

            for ( std::size_t i = 0; i < nin; ++i )
            {
                convolve_add(*net_.input_featuremap(l,i),
                             layer.filter(i,j),
                             *net_.featuremap(l,j));
            }
            transfer_fn_.add_apply(layer.bias(j), *net_.featuremap(l,j));
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

        vec3s out_size = size(grads[0]);
        vec3s in_size  = out_size + layer.filter_size() - vec3s::one;

        for ( std::size_t j = 0; j < nout; ++j )
        {
            transfer_fn_.apply_grad(grads[j], *net_.featuremap(l,j));

            net_.dEdB(l,j) = arma::accu(grads[j]);
        }

        for ( std::size_t i = 0; i < nin; ++i )
        {
            if ( l > 0 )
            {
                igrads.push_back(make_zero_cube<double>(in_size));
            }

            for ( std::size_t j = 0; j < nout; ++j )
            {
                net_.dEdW(l,i,j) = convolve_flipped(*net_.input_featuremap(l,i),
                                                    grads[j]);
                if ( l > 0 )
                {
                    convolve_inverse_add(grads[j], net_.filter(l,i,j),
                                         igrads[i]);
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
        for ( std::size_t i = 0; i < net.num_layers(); ++i )
        {
            if ( net.layer(i).pooling_size() != vec3s::one )
            {
                throw std::logic_error
                    ("simple_network_two doesn't support pooling networks");
            }
        }
    }

    cubes_type forward(const cubes_type& input)
    {
        ZI_ASSERT(input.size()>0);
        ZI_ASSERT(input.size()==net_.num_inputs());

        for ( size_t i = 0; i < net_.num_inputs(); ++i )
        {
            net_.input(i) = pool<double>::get_unique_copy(input[i]);
        }

        for ( std::size_t i = 0; i < net_.num_layers(); ++i )
        {
            forward_layer(i);
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

        net_.apply_grads_serial();
    }

}; // class simple_network_two

}} // namespace zi::znn
