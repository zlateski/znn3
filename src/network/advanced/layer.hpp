#pragma once

#include "../../core/types.hpp"
#include "../../core/cube_utils.hpp"
#include "../../core/diskio.hpp"

#include <vector>
#include <memory>

namespace zi {
namespace znn {

template< class Net >
class layer
{
protected:
    Net*   net_ ;
    size_t id_  ;
    size_t size_;

    std::vector<unique_cube<double>> feature_maps_;

    template<typename Char, typename CharT>
    void write(std::basic_ostream<Char,CharT>& out)
    {
        io::write(out, id_);
        io::write(out, size_);
    }

public:

    template<typename Char, typename CharT>
    layer(Net* net, std::basic_istream<Char,CharT>& in)
        : net_(net)
    {
        io::read(in, id_);
        io::read(in, size_);
        feature_maps_.resize(size_);
    }

    layer(Net* net, size_t id, size_t size)
        : net_(net)
        , id_(id)
        , size_(size)
        , feature_maps_(size)
    {}

    virtual ~layer() {}

    virtual void init(const vec3s&) = 0;
    virtual void forward(size_t, const cube<double>*) = 0;
    virtual void backward(size_t, const cube<double>*) = 0;
    virtual void grad_update() = 0;
    virtual std::string type() const = 0;

    unique_cube<double>& feature_map(size_t n)
    {
        return feature_maps_[n];
    }


    size_t id() const
    {
        return id_;
    }

    size_t size() const
    {
        return size_;
    }

    Net* net()
    {
        return net_;
    }

};


}} // namespace zi::znn
