#pragma once

#include "network_layer.hpp"

namespace zi {
namespace znn {

class layered_network
{
private:
    std::size_t n_inputs_  = 0;
    std::size_t n_outputs_ = 0;
    std::vector<network_layer> layers_;

public:
    template<typename Char, typename CharT>
    void read(std::basic_istream<Char,CharT>& in)
    {
        io::read(in, n_inputs_);

        std::size_t n_layers;
        io::read(in, n_layers);

        layers_.reserve(n_layers);

        for ( std::size_t i = 0; i < n_layers; ++i )
        {
            layers_.emplace_back(in);
        }

        if ( n_layers )
        {
            n_outputs_ = layers_.back().num_outputs();
        }
        else
        {
            n_outputs_ = n_inputs_;
        }
    }

public:
    layered_network()
    {}

    operator bool() const
    {
        return n_inputs_ > 0;
    }

    explicit layered_network(std::size_t n_in)
        : n_inputs_(n_in)
        , n_outputs_(n_in)
    {}

    template<typename Char, typename CharT>
    explicit layered_network(std::basic_istream<Char,CharT>& in)
    {
        read(in);
    }

    layered_network(const layered_network&) = delete;
    layered_network& operator=(const layered_network&) = delete;

    void swap(layered_network& oth)
    {
        if ( this != &oth )
        {
            std::swap(n_inputs_, oth.n_inputs_);
            std::swap(n_outputs_, oth.n_outputs_);
            std::swap(layers_, oth.layers_);
        }
    }

    layered_network& operator=(layered_network&& oth) noexcept
    {
        swap(oth);
        return *this;
    }

    layered_network(layered_network&& oth) noexcept
    {
        swap(oth);
    }


public:
    std::size_t num_inputs() const
    {
        return n_inputs_;
    }

    std::size_t num_outputs() const
    {
        return n_outputs_;
    }

    std::size_t num_layers() const
    {
        return layers_.size();
    }

    network_layer& layer(std::size_t n)
    {
        ZI_ASSERT(n<layers_.size());
        return layers_[n];
    }

    const network_layer& layer(std::size_t n) const
    {
        ZI_ASSERT(n<layers_.size());
        return layers_[n];
    }

    bool operator==(const layered_network& oth) const
    {
        if ( (n_inputs_ != oth.n_inputs_) ||
             (n_outputs_ != oth.n_outputs_) ||
             (layers_.size() != oth.layers_.size() ) )
        {
            return false;
        }
        return std::equal(layers_.begin(), layers_.end(), oth.layers_.begin());
    }

    bool operator!=(const layered_network& oth) const
    {
        return !(*this == oth);
    }

    void pop_layer()
    {
        layers_.pop_back();

        if ( layers_.size() )
        {
            n_outputs_ = layers_.back().num_outputs();
        }
        else
        {
            n_outputs_ = n_inputs_;
        }
    }

    void add_layer( std::size_t n_out,
                    const vec3s& filter_size,
                    const vec3s& pooling_size,
                    double learning_rate)
    {
        layers_.emplace_back(n_outputs_, n_out, filter_size,
                             pooling_size, learning_rate);
        n_outputs_ = n_out;
    }

    void add_layer( std::size_t n_out,
                    const vec3s& filter_size,
                    double learning_rate)
    {
        layers_.emplace_back(n_outputs_, n_out,
                             filter_size, learning_rate);
        n_outputs_ = n_out;
    }

public:
    template<typename Char, typename CharT>
    void write(std::basic_ostream<Char,CharT>& out)
    {
        io::write(out, n_inputs_);
        io::write(out, layers_.size());

        for ( std::size_t i = 0; i < layers_.size(); ++i )
        {
            layers_[i].write(out);
        }
    }

    cube<double>& filter(std::size_t l, std::size_t i, std::size_t j)
    {
        ZI_ASSERT(l<layers_.size());
        return layers_[l].filter(i,j);
    }

    vec3s fov( size_t at_layer = 0 ) const
    {
        if ( at_layer >= layers_.size() )
        {
            return vec3s::one;
        }

        vec3s out_fov = fov(at_layer+1);

        return out_fov * layers_[at_layer].pooling_size()
            + layers_[at_layer].filter_size() - vec3s::one;
    }

}; // class layered_network

}} // namespace zi::znn
