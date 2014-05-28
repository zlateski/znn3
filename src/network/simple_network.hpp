#pragma once

#include <vector>
#include <stdexcept>
#include <functional>

#include "layered_network.hpp"
#include "../transfer_fn/transfer_fn.hpp"
#include "../core/cube_utils.hpp"
#include "../core/cube_pool.hpp"
#include "../convolution/convolve.hpp"

namespace zi {
namespace znn {

class simple_network
{
private:
    typedef std::vector<cube<double>> cubes_type;

private:
    layered_network& net_;
    transfer_fn      transfer_fn_;

    std::vector<cubes_type>                               feature_maps_ ;
    std::vector<std::reference_wrapper<const cubes_type>> ifeature_maps_;

private:
    void forward_layer(std::size_t l, const cubes_type& inputs)
    {
        ZI_ASSERT(l<net_.num_layers());
        ZI_ASSERT(inputs.size()>0);
        ZI_ASSERT(inputs.size()==net_.layer(l).num_inputs());

        ifeature_maps_.emplace_back(inputs);

        network_layer& layer = net_.layer(l);

        std::size_t nin  = layer.num_inputs();
        std::size_t nout = layer.num_outputs();

        vec3s in_size  = size(inputs[0]);
        vec3s out_size = vec3s::one + in_size - layer.filter_size();

        feature_maps_[l].clear();

        for ( std::size_t j = 0; j < nout; ++j )
        {
            feature_maps_[l].push_back(make_zero_cube<double>(out_size));

            for ( std::size_t i = 0; i < nin; ++i )
            {
                convolve_add(inputs[i],
                             layer.filter(i,j),
                             feature_maps_[l][j]);
            }
            transfer_fn_.add_apply(layer.bias(j), feature_maps_[l][j]);
        }
    }

    void backward_layer(std::size_t l, cubes_type& grads,
                        cubes_type& igrads)
    {
        ZI_ASSERT(l<net_.num_layers());
        ZI_ASSERT(ifeature_maps_.size()==l+1);
        ZI_ASSERT(grads.size()>0);
        ZI_ASSERT(grads.size()==net_.layer(l).num_outputs());

        network_layer& layer = net_.layer(l);

        std::size_t nin  = layer.num_inputs();
        std::size_t nout = layer.num_outputs();

        vec3s out_size = size(grads[0]);
        vec3s in_size  = out_size + layer.filter_size() - vec3s::one;

        cube<double> dEdW = make_cube<double>(layer.filter_size());

        igrads.clear();

        double eta = layer.learning_rate();
        const cubes_type& ifeature_maps = ifeature_maps_[l];

        for ( std::size_t j = 0; j < nout; ++j )
        {
            transfer_fn_.apply_grad(grads[j], feature_maps_[l][j]);
            layer.bias(j) -= eta * arma::accu(grads[j]);
        }

        for ( std::size_t i = 0; i < nin; ++i )
        {
            if ( l > 0 )
            {
                igrads.push_back(make_zero_cube<double>(in_size));
            }

            for ( std::size_t j = 0; j < nout; ++j )
            {
                convolve_flipped(ifeature_maps[i], grads[j], dEdW);
                if ( l > 0 )
                {
                    convolve_inverse_add(grads[j], layer.filter(i,j),
                                         igrads[i]);
                }
                dEdW *= eta;
                layer.filter(i,j) -= dEdW;
            }
        }

        ifeature_maps_.pop_back();
    }

public:
    simple_network(layered_network& net, transfer_fn tf)
        : net_(net)
        , transfer_fn_(tf)
        , feature_maps_(net.num_layers())
        , ifeature_maps_()
    {
        ZI_ASSERT(net.num_layers()>0);
        for ( std::size_t i = 0; i < net.num_layers(); ++i )
        {
            if ( net.layer(i).pooling_size() != vec3s::one )
            {
                throw std::logic_error
                    ("simple_network doesn't support pooling networks");
            }
        }
    }

    cubes_type& forward(const cubes_type& input)
    {
        ZI_ASSERT(input.size()>0);
        ZI_ASSERT(input.size()==net_.num_inputs());
        ZI_ASSERT(ifeature_maps_.size()==0);

        forward_layer(0, input);

        for ( std::size_t i = 1; i < net_.num_layers(); ++i )
        {
            forward_layer(i, feature_maps_[i-1]);
        }

        return feature_maps_.back();
    }

    void backward(const cubes_type& grads)
    {
        ZI_ASSERT(grads.size()>0);
        ZI_ASSERT(grads.size()==net_.num_outputs());
        ZI_ASSERT(ifeature_maps_.size()==net_.num_layers());

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
        //net_.apply_grads_serial();
    }


}; // class simple_network

}} // namespace zi::znn
