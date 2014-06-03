#pragma once

#include <cstddef>
#include <cstdlib>
#include <vector>

#include "../core/types.hpp"
#include "../core/cube_utils.hpp"
#include "../core/diskio.hpp"

namespace zi {
namespace znn {

class network_layer
{
public:
    struct random_init {};

private:
    std::size_t n_inputs_     = 0;
    std::size_t n_outputs_    = 0;
    vec3s       filter_size_  = vec3s::one;
    vec3s       pooling_size_ = vec3s::one;
    double      learning_rate_;

    std::vector<std::vector<cube<double>>> filters_;
    std::vector<double>                    biases_;

private:
    template<typename Char, typename CharT>
    void read(std::basic_istream<Char,CharT>& in)
    {
        io::read(in, n_inputs_);
        io::read(in, n_outputs_);
        io::read(in, filter_size_);
        io::read(in, pooling_size_);
        io::read(in, learning_rate_);

        filters_.resize(n_inputs_);
        biases_.resize(n_outputs_);

        for ( std::size_t i = 0; i < n_outputs_; ++i )
        {
            io::read(in, biases_[i]);
        }

        for ( std::size_t i = 0; i < n_inputs_; ++i )
        {
            filters_[i].clear();
            filters_[i].reserve(n_outputs_);
            for ( std::size_t j = 0; j < n_outputs_; ++j )
            {
                filters_[i].emplace_back(filter_size_[0],
                                         filter_size_[1],
                                         filter_size_[2]);
                io::read(in, filters_[i][j]);
            }
        }
    }

public:
    template<typename Char, typename CharT>
    void write(std::basic_ostream<Char,CharT>& out)
    {
        io::write(out, n_inputs_);
        io::write(out, n_outputs_);
        io::write(out, filter_size_);
        io::write(out, pooling_size_);
        io::write(out, learning_rate_);

        for ( std::size_t j = 0; j < n_outputs_; ++j )
        {
            io::write(out, biases_[j]);
        }

        for ( std::size_t i = 0; i < n_inputs_; ++i )
        {
            for ( std::size_t j = 0; j < n_outputs_; ++j )
            {
                io::write(out, filters_[i][j]);
            }
        }
    }


public:
    bool operator==(const network_layer& oth) const
    {
        bool r = (n_inputs_ == oth.n_inputs_)
            && (n_outputs_ == oth.n_outputs_)
            && (filter_size_ == oth.filter_size_)
            && (pooling_size_ == oth.pooling_size_)
            && (learning_rate_ == oth.learning_rate_);

        if ( !r )
        {
            return false;
        }

        for ( std::size_t i = 0; i < n_inputs_; ++i )
        {
            for ( std::size_t j = 0; j < n_outputs_; ++j )
            {
                if ( !equal(filters_[i][j], oth.filters_[i][j]) )
                {
                    return false;
                }
            }
        }

        return true;
    }

    bool operator!=(const network_layer& oth) const
    {
        return !(*this == oth);
    }


public:

    operator bool() const
    {
        return (n_inputs_ > 0) && (n_outputs_ > 0);
    }

    network_layer()
    {}

    network_layer(std::size_t n_in, std::size_t n_out,
                  const vec3s& filter_size,
                  const vec3s& pooling_size,
                  double learning_rate)
        : n_inputs_(n_in)
        , n_outputs_(n_out)
        , filter_size_(filter_size)
        , pooling_size_(pooling_size)
        , learning_rate_(learning_rate)
    {
        filters_.resize(n_inputs_);
        biases_.resize(n_outputs_);

        for ( auto& b: biases_ )
        {
            // todo: better rand generator;
            b = arma::arma_rng_cxx11_instance.randu_val() - 0.5;
            b /= 10;
        }

        for ( std::size_t i = 0; i < n_inputs_; ++i )
        {
            filters_[i].clear();
            filters_[i].reserve(n_outputs_);
            for ( std::size_t j = 0; j < n_outputs_; ++j )
            {
                filters_[i].emplace_back(filter_size_[0],
                                         filter_size_[1],
                                         filter_size_[2]);
                filters_[i][j].randu();
                filters_[i][j] -= 0.5;
                filters_[i][j] /= 10;
            }
        }
    }

    network_layer(std::size_t n_in, std::size_t n_out,
                  const vec3s& filter_size,
                  double learning_rate)
        : network_layer(n_in, n_out, filter_size, vec3s::one, learning_rate)
    {
    }

    template<typename Char, typename CharT>
    explicit network_layer(std::basic_istream<Char,CharT>& in)
    {
        read(in);
    }


    network_layer(const network_layer&) = delete;
    network_layer& operator=(const network_layer&) = delete;

    network_layer& operator=(network_layer&& oth) noexcept
    {
        swap(oth);
        return *this;
    }

    network_layer(network_layer&& oth) noexcept
    {
        swap(oth);
    }

    void swap(network_layer& oth)
    {
        if ( this != &oth )
        {
            std::swap(n_inputs_, oth.n_inputs_);
            std::swap(n_outputs_, oth.n_outputs_);
            std::swap(filter_size_, oth.filter_size_);
            std::swap(pooling_size_, oth.pooling_size_);
            std::swap(learning_rate_, oth.learning_rate_);
            std::swap(filters_, oth.filters_);
            std::swap(biases_, oth.biases_);
        }
    }

    const vec3s& filter_size() const
    {
        return filter_size_;
    }

    const vec3s& pooling_size() const
    {
        return pooling_size_;
    }

    double learning_rate() const
    {
        return learning_rate_;
    }

    double& learning_rate()
    {
        return learning_rate_;
    }

    std::size_t num_inputs() const
    {
        return n_inputs_;
    }

    std::size_t num_outputs() const
    {
        return n_outputs_;
    }

    cube<double>& filter(std::size_t i, std::size_t j)
    {
        ZI_ASSERT(i<n_inputs_);
        ZI_ASSERT(j<n_outputs_);
        return filters_[i][j];
    }

    double& bias(std::size_t i)
    {
        ZI_ASSERT(i<n_outputs_);
        return biases_[i];
    }

}; // class network_layer

}} // namespace zi::znn
