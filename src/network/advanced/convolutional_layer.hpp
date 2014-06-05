#pragma once

#include "layer.hpp"
#include "../../core/cube_pool.hpp"
#include "../../core/sum_of.hpp"
#include "../../transfer_fn/transfer_fn.hpp"

namespace zi {
namespace znn {

template< class Net >
class simple_convolutional_layer: public layer<Net>
{
private:
    typedef layer<Net> base_type;

private:
    size_t num_inputs_ ;
    vec3s  filter_size_;

    std::vector<std::vector<cube<double>>> filters_;
    std::vector<double>                    biases_ ;

    double      eta_;
    transfer_fn transfer_fn_;

    std::vector<cube<double*>>               ifeatures_;
    std::vector<sum_of<unique_cube<double>>> ofeatures_;

    std::vector<double>                      dEdB_ ;
    std::vector<sum_of<unique_cube<double>>> grads_;

    std::vector<std::vector<unique_cube<double>>> dEdW_;

public:
    simple_convolutional_layer( Net* net,
                                size_t id,
                                size_t size,
                                size_t num_inputs,
                                const vec3s& filter_size,
                                double eta,
                                transfer_fn tf)
        : base_type(net, id, size)
        , num_inputs_(num_inputs)
        , filter_size_(filter_size)
        , filters_(num_inputs)
        , biases_(size)
        , eta_(eta)
        , transfer_fn_(tf)
    {
        for ( auto& b: biases_ )
        {
            // todo: better rand generator;
            b = arma::arma_rng_cxx11_instance.randu_val() - 0.5;
            b /= 10;
        }

        for ( std::size_t i = 0; i < num_inputs; ++i )
        {
            filters_[i].clear();
            filters_[i].reserve(size);
            for ( std::size_t j = 0; j < size; ++j )
            {
                filters_[i].emplace_back(filter_size[0],
                                         filter_size[1],
                                         filter_size[2]);
                filters_[i][j].randu();
                filters_[i][j] -= 0.5;
                filters_[i][j] /= 10;
            }
        }
    }

    simple_convolutional_layer(Net* net, io::istream& in)
        : base_type(net, in)
    {
    }

    void init(const vec3s& s = vec3s::one) override
    {
        base_type::net()->init_done(base_type::id(), s);
    }

    void forward(size_t n, const cube<double>* c) override
    {
        base_type::net()->forward_done(base_type::id(), n, c);
    }

    void backward(size_t n, const cube<double>* c) override
    {
        base_type::net()->backward_done(base_type::id(), n, c);
    }

    std::string type() const override
    {
        return "simple_convolutional";
    }

}; // class simple_convolutional_layer


}} // namespace zi::znn
