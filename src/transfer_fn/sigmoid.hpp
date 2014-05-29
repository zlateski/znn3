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

struct sigmoid_for_logreg
{
    double operator()(double x) const
    {
        return static_cast<double>(1) / (static_cast<double>(1) + std::exp(-x));
    }

    double grad(double) const
    {
        return 1;
    }
};

}} // namespace zi::znn
