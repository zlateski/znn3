#pragma once

#include <cmath>

namespace zi {
namespace znn {

struct sigmoid
{
    double operator()(double x) const
    {
        return static_cast<double>(1) / (static_cast<double>(1) + std::exp(-x));
    }

    double grad(double f) const
    {
        return f * (static_cast<double>(1) - f);
    }

}; // struct sigmoid

}} // namespace zi::znn
