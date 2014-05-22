#pragma once

#include <cmath>

namespace zi {
namespace znn {

struct hyperbolic_tangent
{
private:
    double a_;
    double b_;
    double b_over_a_;

public:
    hyperbolic_tangent(double a = 1.7159, double b = 0.6666)
        : a_(a), b_(b), b_over_a_(b/a)
    {}

    double operator()(double x) const
    {
        return a_ * std::tanh(b_ * x);
    }

    double grad(double f) const
    {
        return b_over_a_ * ( a_ - f ) * ( a_ + f );
    }

}; // struct hyperbolic_tangent

}} // namespace zi::znn
