#pragma once

#include <algorithm>

namespace zi {
namespace znn {

struct rectify_linear
{
    double operator()(double x) const
    {
        return std::max(x, static_cast<double>(0));
    }

    double grad(double f) const
    {
        return (f > 0) ? 1 : 0;
    }

}; // struct rectify_linear

}} // namespace zi::znn
