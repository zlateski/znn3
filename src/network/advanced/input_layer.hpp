#pragma once

#include "layer.hpp"
#include "../../core/cube_pool.hpp"

namespace zi {
namespace znn {

template< class Net >
class input_layer: virtual public layer<Net>
{
private:
    typedef layer<Net> base_type;

public:
    virtual ~input_layer() {}

    input_layer(Net* net, size_t id, size_t size)
        : base_type(net, id, size)
    {}

    input_layer(Net* net, io::istream& in)
        : base_type(net, in)
    {}

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

    void grad_update() override
    {
        base_type::net()->grad_update_done(base_type::id());
    }

    std::string type() const override
    {
        return "input";
    }

}; // class input_layer


}} // namespace zi::znn
