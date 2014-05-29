#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <functional>
#include <atomic>

#include <zi/async.hpp>

#include "layered_network.hpp"
#include "layered_network_data.hpp"
#include "../transfer_fn/transfer_fn.hpp"
#include "../core/cube_utils.hpp"
#include "../core/fft.hpp"
#include "../core/carrier.hpp"
#include "../convolution/sparse_convolve.hpp"
#include "../core/cube_pool.hpp"
#include "../pooling/pooling_filter_2.hpp"


// TODO: Fix transfer function hack

namespace zi {
namespace znn {


class parallel_network_layer
{
public:
    virtual ~parallel_network_layer() {};
    virtual void init(const vec3s&) = 0;
    virtual void run_forward(size_t) = 0;
    virtual void run_backward(size_t, unique_cube<double>&) = 0;
};

template< class Net >
class parallel_network_layer_direct
    : public parallel_network_layer
{
private:
    struct input_perceptron_data
    {
        unique_cube<double>   grad           ;
        std::mutex            mutex          ;
        size_t                received = 0   ;
    };

    struct output_perceptron_data
    {
        std::mutex            mutex;
        size_t                received = 0;
        unique_cube<uint32_t> pooling_indices;
    };

private:
    typedef parallel_network_layer_direct<Net> this_type   ;
    typedef Net                                network_type;

private:
    network_type&         network_    ;
    layered_network_data& data_       ;
    size_t                layer_no_   ;
    transfer_fn&          transfer_fn_;

    vec3s                 sparsness = vec3s::one;

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
    {
        // if ( layer_no_ == data_.num_layers() - 1 )
        // {
        //     transfer_fn_ = make_transfer_fn<sigmoid_for_logreg>();
        // }
    }

private:
    void forward_filter(size_t i, size_t o)
    {
        const unique_cube<double>& f = data_.input_featuremap(layer_no_, i);

        ZI_ASSERT(inputs_[i].received==0);
        ZI_ASSERT(i<inputs_.size());
        ZI_ASSERT(o<outputs_.size());
        ZI_ASSERT(f);

        // Convolve the featuremap with the appropriate filter

        unique_cube<double> convolved =
            sparse_convolve(*f, data_.filter(layer_no_,i,o), sparsness);

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

                if ( data_.pooling_size(layer_no_) != vec3s::one )
                {
                    auto pooled =
                        pooling_filter_2(*fout, std::greater<double>(),
                                         data_.pooling_size(layer_no_),
                                         sparsness);

                    fout = std::move(pooled.first);
                    perc.pooling_indices = std::move(pooled.second);
                }

                zi::async::async(&Net::forward_done, &network_, layer_no_, o);
            }
        }
    }


    void backward_filter(size_t l, size_t r, unique_cube<double>& g)
    {
        ZI_ASSERT(l<inputs_.size());
        ZI_ASSERT(r<outputs_.size());

        unique_cube<double>& ifmap = data_.input_featuremap(layer_no_,l);
        unique_cube<double>& dEdW  = data_.dEdW(layer_no_,l,r);

        input_perceptron_data& perceptron = inputs_[l];

        // This is where we would implement momentum
        dEdW = sparse_convolve_flipped(*ifmap, *g, sparsness);

        if ( layer_no_ > 0 )
        {
            unique_cube<double> gadd =
                sparse_convolve_inverse(*g, data_.filter(layer_no_,l,r),
                                        sparsness);

            while (1)
            {
                unique_cube<double> old;

                {
                    guard gd(perceptron.mutex);

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
                                      l, std::ref(perceptron.grad));
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
                                  std::ref(perceptron.grad));
            }
        }
    }


public:

    void init( const vec3s& sparse )
    {
        sparsness = sparse;
        network_.init_done(layer_no_, sparse * data_.pooling_size(layer_no_));
    }

    void run_forward(size_t pno)
    {
        // shouldn't have to lock here
        ZI_ASSERT(pno<inputs_.size());
        ZI_ASSERT(inputs_[pno].received==0);

        for ( size_t i = 0; i < outputs_.size(); ++i )
        {
            zi::async::async_priority(layer_no_ * 1000 + pno,
                                      &this_type::forward_filter, this,
                                      pno, i);
        }
    }

    void run_backward(size_t perceptron_no, unique_cube<double>& g)
    {
        ZI_ASSERT(perceptron_no<outputs_.size());
        ZI_ASSERT(outputs_[perceptron_no].received==0);

        transfer_fn_.apply_grad(*g, *data_.featuremap(layer_no_,
                                                      perceptron_no));


        data_.dEdB(layer_no_, perceptron_no) = arma::accu(*g);

        if ( data_.pooling_size(layer_no_) != vec3s::one )
        {
            g = pooling_filter_2_bprop(
                *g, *outputs_[perceptron_no].pooling_indices,
                data_.pooling_size(layer_no_),
                sparsness);
        }

        for ( size_t i = 0; i < inputs_.size(); ++i )
        {
            zi::async::async_priority(2000000 - layer_no_*1000 - perceptron_no,
                                      &this_type::backward_filter, this, i,
                                      perceptron_no, std::ref(g));
        }
    }

};


template< class Net >
class parallel_network_layer_fft
    : public parallel_network_layer
{
private:
    struct input_perceptron_data
    {
        unique_cube<double>   grad           ;
        std::mutex            mutex          ;
        size_t                received = 0   ;
        unique_cube<complex>  featuremap_fft ;
        unique_cube<complex>  grad_fft       ;
        std::vector<unique_cube<complex>> w_fft;
    };

    struct output_perceptron_data
    {
        std::mutex            mutex;
        size_t                received = 0;
        unique_cube<uint32_t> pooling_indices;
        unique_cube<complex>  featuremap_fft ;
        unique_cube<complex>  grad_fft       ;
        std::atomic<int>      grad_fft_to_send;
    };

private:
    typedef parallel_network_layer_fft<Net> this_type   ;
    typedef Net                             network_type;

private:
    network_type&         network_    ;
    layered_network_data& data_       ;
    size_t                layer_no_   ;
    transfer_fn&          transfer_fn_;

    vec3s                 sparsness        = vec3s::one;
    vec3s                 real_filter_size = vec3s::one;

    std::vector<input_perceptron_data>  inputs_ ;
    std::vector<output_perceptron_data> outputs_;

public:
    parallel_network_layer_fft(network_type& net, size_t layer_no)
        : network_(net)
        , data_(net.data())
        , layer_no_(layer_no)
        , transfer_fn_(net.transfer_function())
        , inputs_(data_.layer(layer_no).num_inputs())
        , outputs_(data_.layer(layer_no).num_outputs())
    {
        // if ( layer_no_ == data_.num_layers() - 1 )
        // {
        //     transfer_fn_ = make_transfer_fn<sigmoid_for_logreg>();
        // }
    }

private:
    void forward_filter(size_t i, size_t o)
    {
        // f is the reference to the input featuremap, we mostly need it
        // to get the size of the featuremap for the fft calls

        const unique_cube<double>& f = data_.input_featuremap(layer_no_, i);

        ZI_ASSERT(inputs_[i].featuremap_fft);
        ZI_ASSERT(inputs_[i].received==0);
        ZI_ASSERT(i<inputs_.size());
        ZI_ASSERT(o<outputs_.size());
        ZI_ASSERT(f);

        input_perceptron_data&  iperc = inputs_[i];
        output_perceptron_data& operc = outputs_[o];


        // Compute the filter's transform, in the case it's not already there
        // Or if the input size had changed.
        // WARNING - there's edge cases where this comparison might not work!

        // if ( (!iperc.w_fft[o]) ||
        //      (size(*iperc.w_fft[o]) != fft_complex_size(*f) ) )
        {
            iperc.w_fft[o] =
                fftw::forward_pad( data_.filter(layer_no_,i,o),
                                   sparsness, size(*f) );
        }

        // Convolve (pairwise multiplication of the fft transforms)
        // the featuremap with the appropriate filter

        unique_cube<complex> to_add =
            pool<complex>::get_unique_copy(*iperc.featuremap_fft);

        pairwise_mult(*to_add, *inputs_[i].w_fft[o]);

        // Update the output perceptron

        while (1)
        {
            unique_cube<complex> old;
            {
                guard g(operc.mutex);
                if ( operc.received == 0 )
                {
                    ++operc.received;
                    operc.featuremap_fft = std::move(to_add);
                    break;
                }
                else
                {
                    if ( operc.featuremap_fft )
                    {
                        old = std::move(operc.featuremap_fft);
                    }
                    else
                    {
                        ++operc.received;
                        operc.featuremap_fft = std::move(to_add);
                        break;
                    }
                }
            }
            *to_add += *old;
        }

        // Check if this is the last perceptron to send values to the outgoing
        // perceptron. In which case we are ready to process the target
        // perceptron

        {
            guard g(operc.mutex);
            if ( operc.received == inputs_.size() )
            {
                operc.received = 0;

                unique_cube<double>& fout = data_.featuremap(layer_no_, o);

                auto x = fftw::backward( *operc.featuremap_fft, size(*f) );
                operc.featuremap_fft.reset();

                vec3s out_f_size = size(*f) + vec3s::one - real_filter_size;

                fout = crop_right(*x, out_f_size);

                *fout /= x->n_elem;

                transfer_fn_.add_apply(data_.bias(layer_no_,o), *fout);

                if ( data_.pooling_size(layer_no_) != vec3s::one )
                {
                    auto pooled =
                        pooling_filter_2(*fout, std::greater<double>(),
                                         data_.pooling_size(layer_no_),
                                         sparsness);

                    fout = std::move(pooled.first);
                    operc.pooling_indices = std::move(pooled.second);
                }

                zi::async::async(&Net::forward_done, &network_, layer_no_, o);
            }
        }
    }


    void backward_filter(size_t l, size_t r)
    {
        const unique_cube<double>& f = data_.input_featuremap(layer_no_, l);

        ZI_ASSERT(l<inputs_.size());
        ZI_ASSERT(r<outputs_.size());

        input_perceptron_data&  iperc = inputs_[l];
        output_perceptron_data& operc = outputs_[r];

        vec3s s = size(*f);

        auto dEdW_fft = pool<complex>::get_unique_copy(*operc.grad_fft);

        pairwise_mult(*dEdW_fft, *iperc.featuremap_fft);

        unique_cube<double>& dEdW = data_.dEdW(layer_no_,l,r);

        dEdW = fftw::backward(*dEdW_fft, s);

        dEdW = sparse_implode_flip( *dEdW, size(data_.filter(layer_no_,l,r)),
                                    sparsness );


        *dEdW /= s[0]*s[1]*s[2];

        unique_cube<complex> to_add;

        if ( layer_no_ > 0 )
        {
            to_add = pool<complex>::get_unique_copy(*operc.grad_fft);
            pairwise_mult(*to_add, *inputs_[l].w_fft[r]);
        }

        if ( --operc.grad_fft_to_send == 0 )
        {
            operc.grad_fft.reset();
        }

        if ( layer_no_ > 0 )
        {
            while (1)
            {
                unique_cube<complex> old;

                {
                    guard gd(iperc.mutex);

                    if ( iperc.received == 0 )
                    {
                        ++iperc.received;
                        iperc.grad_fft = std::move(to_add);
                        break;
                    }
                    else
                    {
                        if ( iperc.grad_fft )
                        {
                            old = std::move(iperc.grad_fft);
                        }
                        else
                        {
                            iperc.grad_fft = std::move(to_add);
                            ++iperc.received;
                            break;
                        }
                    }
                }

                *to_add += *old;
            }

            {
                guard g(iperc.mutex);
                if ( iperc.received == outputs_.size() )
                {
                    iperc.received = 0;

                    iperc.grad = fftw::backward(*iperc.grad_fft, size(*f));
                    iperc.grad_fft.reset();

                    flip_dims(*iperc.grad);
                    *iperc.grad /= iperc.grad->n_elem;


                    zi::async::async( &Net::backward_done, &network_,layer_no_,
                                      l, std::ref(iperc.grad));
                }
            }
        }
        else
        {
            guard g(iperc.mutex);
            ++iperc.received;
            if ( iperc.received == outputs_.size() )
            {
                iperc.received = 0;
                zi::async::async( &Net::backward_done, &network_,layer_no_, l,
                                  std::ref(iperc.grad));
            }
        }
    }


public:

    void init( const vec3s& sparse )
    {
        sparsness = sparse;
        real_filter_size = (data_.filter_size(layer_no_) - vec3s::one)
            * sparse + vec3s::one;

        network_.init_done(layer_no_, sparse * data_.pooling_size(layer_no_));
    }

    void run_forward(size_t pno)
    {
        // shouldn't have to lock here
        ZI_ASSERT(pno<inputs_.size());
        ZI_ASSERT(inputs_[pno].received==0);


        // The input featuremap tranforms are saved in order to calculate dEdW.

        inputs_[pno].featuremap_fft =
            fftw::forward_copy(*data_.input_featuremap(layer_no_, pno));

        // Check the filter transforms. We want to cache them, but we
        // clear them on the gradient update by clearing the whole vector.
        // This will heep the cache or create empty pointers otherwise.

        inputs_[pno].w_fft.resize(outputs_.size());

        // Envoke a task for each of the perceptron' filters

        for ( size_t i = 0; i < outputs_.size(); ++i )
        {
            zi::async::async_priority(layer_no_ * 1000 + pno,
                                      &this_type::forward_filter, this,
                                      pno, i);
        }
    }

    void run_backward(size_t perceptron_no, unique_cube<double>& g)
    {
        ZI_ASSERT(perceptron_no<outputs_.size());
        ZI_ASSERT(outputs_[perceptron_no].received==0);

        output_perceptron_data& operc = outputs_[perceptron_no];

        // This is the featuremap of the forward pass after applying
        // the pooling filter (if any)

        const unique_cube<double>& f =
            data_.featuremap(layer_no_, perceptron_no);

        transfer_fn_.apply_grad(*g, *f);

        data_.dEdB(layer_no_, perceptron_no) = arma::accu(*g);

        // If sparse, decompress the sparsed gradient

        if ( data_.pooling_size(layer_no_) != vec3s::one )
        {
            g = pooling_filter_2_bprop( *g, *operc.pooling_indices,
                                        data_.pooling_size(layer_no_),
                                        sparsness);
        }

        flip_dims(*g);

        // might be able to get rid of this
        vec3s in_f_size = size(*g) + real_filter_size - vec3s::one;
        ZI_ASSERT(size(*data_.input_featuremap(layer_no_,0))==in_f_size);

        operc.grad_fft = fftw::forward_pad(*g, in_f_size);
        operc.grad_fft_to_send = inputs_.size();

        // clear the memory for grad
        g.reset();

        for ( size_t i = 0; i < inputs_.size(); ++i )
        {
            zi::async::async_priority(2000000 - layer_no_*1000 - perceptron_no,
                                      &this_type::backward_filter, this, i,
                                      perceptron_no);
        }
    }

};



class parallel_network
{
private:
    typedef parallel_network_layer_direct<parallel_network> direct_layer_type;
    typedef parallel_network_layer_fft<parallel_network>    fft_layer_type;
    typedef std::unique_ptr<parallel_network_layer>         layer_ptr ;
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

    void do_backward(size_t i, unique_cube<double>& g)
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
            // if ( i % 2 )
            //     layers_[i] = layer_ptr(new direct_layer_type(*this, i));
            // else
                layers_[i] = layer_ptr(new fft_layer_type(*this, i));
        }
        layers_[0]->init(vec3s::one);
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

        std::vector<unique_cube<double>> my_grads(grads.size());

        for ( size_t i = 0; i < grads.size(); ++i )
        {
            my_grads[i] = pool<double>::get_unique_copy(grads[i]);

            zi::async::async(&parallel_network::do_backward,
                             this, i, std::ref(my_grads[i]));
        }

        waiter_.wait();

    }

    void grad_update()
    {
        net_.apply_grads();
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

    void init_done(size_t l, const vec3s& sparse)
    {
        if ( l < net_.num_layers() - 1 )
        {
            layers_[l+1]->init(sparse);
        }
    }

    void backward_done(size_t l, size_t p, unique_cube<double>& c)
    {
        if ( l > 0 )
        {
            layers_[l-1]->run_backward(p, c);
        }
        else
        {
            waiter_.one_done();
        }
    }

    vec3s fov()
    {
        return net_.fov();
    }

}; // class parallel_network

}} // namespace zi::znn
