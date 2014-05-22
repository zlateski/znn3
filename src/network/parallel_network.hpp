#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <functional>

#include <zi/async.hpp>

#include "layered_network.hpp"
#include "layered_network_data.hpp"
#include "../transfer_fn/transfer_fn.hpp"
#include "../core/cube_utils.hpp"
#include "../core/carrier.hpp"
#include "../convolution/convolve.hpp"
#include "../core/cube_pool.hpp"


namespace zi {
namespace znn {


class parallel_network_layer
{
public:
    virtual void run_forward(size_t) = 0;
    virtual void run_backward(size_t, cube<double>&) = 0;
};

template< class Net >
class parallel_network_layer_direct
    : public parallel_network_layer
{
private:
    struct input_perceptron_data
    {
        unique_cube<double>  grad ;
        std::mutex           mutex;
        size_t               received = 0;
    };

    struct output_perceptron_data
    {
        std::mutex       mutex;
        size_t           received = 0;
    };

private:
    typedef parallel_network_layer_direct<Net> this_type   ;
    typedef Net                                network_type;

private:
    network_type&         network_    ;
    layered_network_data& data_       ;
    size_t                layer_no_   ;
    transfer_fn&          transfer_fn_;

    std::vector<input_perceptron_data>  inputs_ ;
    std::vector<output_perceptron_data> outputs_;

public:
    parallel_network_layer_direct(network_type& net, size_t layer_no)
        : network_(net)
        , data_(net.data())
        , layer_no_(layer_no)
        , transfer_fn_(net.transfer_function())
        , inputs_(data_.layer(layer_no).num_inputs())
        , outputs_(data_.layer(layer_no).num_outputs())
    {}

    void forward_filter(size_t i, size_t o)
    {
        const unique_cube<double>& f = data_.input_featuremap(layer_no_, i);

        ZI_ASSERT(inputs_[i].received==0);
        ZI_ASSERT(i<inputs_.size());
        ZI_ASSERT(o<outputs_.size());
        ZI_ASSERT(f);

        // Convolve the featuremap with the appropriate filter

        unique_cube<double> convolved =
            convolve(*f, data_.filter(layer_no_,i,o));

        unique_cube<double>&    fout = data_.featuremap(layer_no_, o);
        output_perceptron_data& perc = outputs_[o];

        while (1)
        {
            unique_cube<double> old;
            {
                guard g(perc.mutex);
                if ( perc.received == 0 )
                {
                    ++perc.received;
                    fout = std::move(convolved);
                    break;
                }
                else
                {
                    if ( fout )
                    {
                        old = std::move(fout);
                    }
                    else
                    {
                        ++perc.received;
                        fout = std::move(convolved);
                        break;
                    }
                }
            }
            *convolved += *old;
        }

        {
            guard g(perc.mutex);
            if ( perc.received == inputs_.size() )
            {
                perc.received = 0;
                transfer_fn_.add_apply(data_.bias(layer_no_,o), *fout);
                zi::async::async(&Net::forward_done, &network_, layer_no_, o);
            }
        }
    }


    void backward_filter(size_t l, size_t r, cube<double>& g)
    {
        ZI_ASSERT(l<inputs_.size());
        ZI_ASSERT(r<outputs_.size());

        unique_cube<double>& ifmap = data_.input_featuremap(layer_no_,l);
        unique_cube<double>& dEdW  = data_.dEdW(layer_no_,l,r);

        input_perceptron_data& perceptron = inputs_[l];

        // This is where we would implement momentum
        dEdW = convolve_flipped(*ifmap, g);

        if ( layer_no_ > 0 )
        {
            unique_cube<double> gadd =
                convolve_inverse(g, data_.filter(layer_no_,l,r));

            while (1)
            {
                unique_cube<double> old;

                {
                    guard g(perceptron.mutex);

                    if ( perceptron.received == 0 )
                    {
                        ++perceptron.received;
                        perceptron.grad = std::move(gadd);
                        break;
                    }
                    else
                    {
                        if ( perceptron.grad )
                        {
                            old = std::move(perceptron.grad);
                        }
                        else
                        {
                            perceptron.grad = std::move(gadd);
                            ++perceptron.received;
                            break;
                        }
                    }
                }

                *gadd += *old;
            }

            {
                guard g(perceptron.mutex);
                if ( perceptron.received == outputs_.size() )
                {
                    perceptron.received = 0;
                    zi::async::async( &Net::backward_done, &network_,layer_no_,
                                      l, std::cref(perceptron.grad));
                }
            }
        }
        else
        {
            guard g(perceptron.mutex);
            ++perceptron.received;
            if ( perceptron.received == outputs_.size() )
            {
                perceptron.received = 0;
                zi::async::async( &Net::backward_done, &network_,layer_no_, l,
                                  std::cref(perceptron.grad));
            }
        }
    }


public:

    void run_forward(size_t pno)
    {
        // shouldn't have to lock here
        ZI_ASSERT(pno<inputs_.size());
        ZI_ASSERT(inputs_[pno].received==0);

        for ( size_t i = 0; i < outputs_.size(); ++i )
        {
            zi::async::async_priority(layer_no_ * 1000 + pno,
                                      &this_type::forward_filter, this, pno, i);
        }
    }

    void run_backward(size_t perceptron_no, cube<double>& g)
    {
        ZI_ASSERT(perceptron_no<outputs_.size());
        ZI_ASSERT(outputs_[perceptron_no].received==0);

        transfer_fn_.apply_grad(g, *data_.featuremap(layer_no_, perceptron_no));

        data_.dEdB(layer_no_, perceptron_no) = arma::accu(g);

        for ( size_t i = 0; i < inputs_.size(); ++i )
        {
            zi::async::async_priority(2000000 - layer_no_*1000 - perceptron_no,
                                      &this_type::backward_filter, this, i,
                                      perceptron_no, std::ref(g));
        }
    }

};



class parallel_network
{
private:
    typedef parallel_network_layer_direct<parallel_network> layer_type;
    typedef std::unique_ptr<layer_type>                     layer_ptr ;
    typedef std::vector<cube<double>>                       cubes_type;

private:
    layered_network_data&  net_;
    transfer_fn            transfer_fn_;
    std::vector<layer_ptr> layers_;
    waiter                 waiter_;

private:
    void do_forward(size_t i, const cube<double>& f)
    {
        net_.input(i) = pool<double>::get_unique_copy(f);
        layers_.front()->run_forward(i);
    }

    void do_backward(size_t i, cube<double>& g)
    {
        layers_.back()->run_backward(i, g);
    }

public:

    transfer_fn& transfer_function()
    {
        return transfer_fn_;
    }

    layered_network_data& data()
    {
        return net_;
    }

    parallel_network(layered_network_data& net, transfer_fn tf)
        : net_(net)
        , transfer_fn_(tf)
        , layers_(net.num_layers())
    {
        for ( size_t i = 0; i < layers_.size(); ++i )
        {
            layers_[i] = layer_ptr(new layer_type(*this, i));
        }
    }

    cubes_type forward(const cubes_type& input)
    {
        ZI_ASSERT(input.size()>0);
        ZI_ASSERT(input.size()==net_.num_inputs());

        waiter_.set(net_.num_outputs());

        for ( size_t i = 0; i < input.size(); ++i )
        {
            zi::async::async(&parallel_network::do_forward,
                             this, i, std::ref(input[i]));
        }

        waiter_.wait();

        cubes_type ret(net_.num_outputs());
        for ( size_t i = 0; i < net_.num_outputs(); ++i )
        {
            ret[i] = *net_.output(i);
        }

        return ret;
    }

    void backward(const cubes_type& grads)
    {
        ZI_ASSERT(grads.size()>0);
        ZI_ASSERT(grads.size()==net_.num_outputs());

        waiter_.set(net_.num_inputs());

        cubes_type my_grads = grads;

        for ( size_t i = 0; i < grads.size(); ++i )
        {
            zi::async::async(&parallel_network::do_backward,
                             this, i, std::ref(my_grads[i]));
        }

        waiter_.wait();

        net_.apply_grads_serial();
    }

    void forward_done(size_t l, size_t p)
    {
        if ( l < net_.num_layers() - 1 )
        {
            layers_[l+1]->run_forward(p);
        }
        else
        {
            waiter_.one_done();
        }
    }

    void backward_done(size_t l, size_t p, const unique_cube<double>& c)
    {
        if ( l > 0 )
        {
            layers_[l-1]->run_backward(p, *c);
        }
        else
        {
            waiter_.one_done();
        }
    }


}; // class parallel_network

}} // namespace zi::znn
