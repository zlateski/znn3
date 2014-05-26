#pragma once

#include <zi/async.hpp>

#include "layered_network.hpp"
#include "../core/waiter.hpp"
#include "../core/cube_pool.hpp" // for unuque_cube


namespace zi {
namespace znn {


class layered_network_data
{
private:
    struct layer_data
    {
        std::vector<unique_cube<double>>               featuremaps;
        std::vector<double>                            dEdB;
        std::vector<std::vector<unique_cube<double>>>  dEdW;
    };

private:
    layered_network&                 network_;
    std::vector<layer_data>          layer_data_;
    std::vector<unique_cube<double>> inputs_;

    size_t num_perceptrons_ = 0;
    size_t num_filters_     = 0;
    size_t num_layers_      = 0;

    void init()
    {
        num_perceptrons_ = 0;
        num_filters_ = 0;
        num_layers_ = network_.num_layers();
        layer_data_.resize(num_layers_);

        inputs_.resize(network_.num_inputs());

        for ( size_t i = 0; i < layer_data_.size(); ++i )
        {
            layer_data_[i].featuremaps.resize(network_.layer(i).num_outputs());
            layer_data_[i].dEdB.resize(network_.layer(i).num_outputs());
            layer_data_[i].dEdW.resize(network_.layer(i).num_inputs());

            num_perceptrons_ += network_.layer(i).num_outputs();

            for ( auto& l: layer_data_[i].dEdW )
            {
                l.resize(network_.layer(i).num_outputs());
                num_filters_ += network_.layer(i).num_outputs();
            }
        }
    }

    void apply_grad(size_t layer, size_t j, waiter& w)
    {
        bias(layer,j) -= learning_rate(layer) * dEdB(layer,j);

        for ( size_t i = 0; i < layer_data_[layer].dEdW.size(); ++i )
        {
            ZI_ASSERT(layer_data_[layer].dEdW[i][j]);

            *dEdW(layer,i,j) *= learning_rate(layer);
            filter(layer,i,j) -= *dEdW(layer,i,j);
        }

        w.one_done();
    }

    void apply_grad_serial(size_t layer, size_t i, size_t j)
    {
        ZI_ASSERT(layer_data_[layer].dEdW[i][j]);

        *dEdW(layer,i,j) *= learning_rate(layer);
        filter(layer,i,j) -= *dEdW(layer,i,j);
    }

public:
    layered_network_data( layered_network& network )
        : network_(network)
    {
        init();
    }

    void clear()
    {
        layer_data_.clear();
        inputs_.clear();
        init();
    }

    unique_cube<double>& input(size_t i)
    {
        return inputs_[i];
    }

    unique_cube<double>& output(size_t i)
    {
        return layer_data_.back().featuremaps[i];
    }

    std::size_t num_inputs() const
    {
        return network_.num_inputs();
    }

    std::size_t num_outputs() const
    {
        return network_.num_outputs();
    }

    std::size_t num_layers() const
    {
        return layer_data_.size();
    }

    network_layer& layer(std::size_t n)
    {
        return network_.layer(n);
    }

    cube<double>& filter(std::size_t l, std::size_t i, std::size_t j)
    {
        return network_.layer(l).filter(i,j);
    }

    double learning_rate(size_t l) const
    {
        return network_.layer(l).learning_rate();
    }

    double& learning_rate(size_t l)
    {
        return network_.layer(l).learning_rate();
    }

    double& dEdB(size_t l, size_t p)
    {
        return layer_data_[l].dEdB[p];
    }

    unique_cube<double>& featuremap(size_t l, size_t p)
    {
        return layer_data_[l].featuremaps[p];
    }

    unique_cube<double>& input_featuremap(size_t l, size_t p)
    {
        return (l == 0) ? inputs_[p] : layer_data_[l-1].featuremaps[p];
    }

    unique_cube<double>& dEdW(size_t l, size_t i, size_t j)
    {
        return layer_data_[l].dEdW[i][j];
    }

    double& bias(size_t l, size_t i)
    {
        return network_.layer(l).bias(i);
    }

    void apply_grads()
    {
        waiter w(num_perceptrons_);

        for ( size_t l = 0; l < num_layers_; ++l )
        {
            for ( size_t j = 0; j < layer_data_[l].dEdB.size(); ++j )
            {
                zi::async::async(&layered_network_data::apply_grad,
                                     this, l, j, std::ref(w));
            }
        }

        w.wait();
    }

    void apply_grads_serial()
    {
        for ( size_t l = 0; l < num_layers_; ++l )
        {
            for ( size_t j = 0; j < layer_data_[l].dEdB.size(); ++j )
            {
                bias(l,j) -= learning_rate(l) * dEdB(l,j);
                for ( size_t i = 0; i < layer_data_[l].dEdW.size(); ++i )
                {
                    apply_grad_serial(l, i, j);
                }
            }
        }
    }


}; // class layered_network_data

}} // namespace zi::znn
