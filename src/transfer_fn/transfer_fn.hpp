#pragma once

#include <memory>
#include <algorithm>
#include <cmath>
#include "../core/types.hpp"

#include "sigmoid.hpp"
#include "rectify_linear.hpp"
#include "hyperbolic_tangent.hpp"

namespace zi {
namespace znn {

class transfer_fn_interface
{
public:
    virtual void apply_grad( cube<double>& /* dEdF */,
                             const cube<double>& /* F */) const = 0;

    virtual void apply( cube<double>& ) const = 0;

    virtual void add_apply( double, cube<double>& ) const = 0;

    virtual double operator()( double ) const = 0;

    virtual double grad( double ) const = 0;
};

template< class Fn >
class transfer_fn_wrapper
    : public transfer_fn_interface
{
private:
    Fn f_;

    transfer_fn_wrapper(const transfer_fn_wrapper&) = delete;
    transfer_fn_wrapper(transfer_fn_wrapper&&) = delete;
    transfer_fn_wrapper& operator=(const transfer_fn_wrapper&) = delete;
    transfer_fn_wrapper& operator=(transfer_fn_wrapper&&) = delete;

public:
    transfer_fn_wrapper(Fn f)
        : f_(f)
    {}

    void apply_grad( cube<double>& dEdF,
                     const cube<double>& F) const override
    {
        ZI_ASSERT(dEdF.n_elem==F.n_elem);
        double* r = dEdF.memptr();
        const double* f = F.memptr();
        for ( std::size_t i = 0; i < dEdF.n_elem; ++i )
        {
            r[i] *= f_.grad(f[i]);
        }
    }

    void apply( cube<double>& F ) const override
    {
        double* f = F.memptr();
        for ( std::size_t i = 0; i < F.n_elem; ++i )
        {
            f[i] = f_(f[i]);
        }
    }

    void add_apply( double c, cube<double>& F ) const override
    {
        double* f = F.memptr();
        for ( std::size_t i = 0; i < F.n_elem; ++i )
        {
            f[i] = f_(f[i]+c);
        }
    }

    double operator()(double x) const override
    {
        return f_(x);
    }

    double grad(double f) const override
    {
        return f_.grad(f);
    }
};

template< class Fn, class Grad >
class transfer_fn_wrapper2
    : public transfer_fn_interface
{
private:
    Fn   f_;
    Grad g_;

    transfer_fn_wrapper2(const transfer_fn_wrapper2&) = delete;
    transfer_fn_wrapper2(transfer_fn_wrapper2&&) = delete;
    transfer_fn_wrapper2& operator=(const transfer_fn_wrapper2&) = delete;
    transfer_fn_wrapper2& operator=(transfer_fn_wrapper2&&) = delete;

public:
    transfer_fn_wrapper2(Fn f, Grad g)
        : f_(f), g_(g)
    {}

    void apply_grad( cube<double>& dEdF,
                     const cube<double>& F) const override
    {
        ZI_ASSERT(dEdF.n_elem==F.n_elem);
        double* r = dEdF.memptr();
        const double* f = F.memptr();
        for ( std::size_t i = 0; i < dEdF.n_elem; ++i )
        {
            r[i] *= g_(f[i]);
        }
    }

    void apply( cube<double>& F ) const override
    {
        double* f = F.memptr();
        for ( std::size_t i = 0; i < F.n_elem; ++i )
        {
            f[i] = f_(f[i]);
        }
    }

    void add_apply( double c, cube<double>& F ) const override
    {
        double* f = F.memptr();
        for ( std::size_t i = 0; i < F.n_elem; ++i )
        {
            f[i] = f_(f[i]+c);
        }
    }

    double operator()(double x) const override
    {
        return f_(x);
    }

    double grad(double f) const override
    {
        return g_(f);
    }
};


struct transfer_fn
{
private:
    std::shared_ptr<transfer_fn_interface> impl_ = nullptr;

public:
    transfer_fn()
    {}

    template<class F>
    transfer_fn(F f)
        : impl_(new transfer_fn_wrapper<F>(f))
    {}

    template<class F, class G>
    transfer_fn(F f, G g)
        : impl_(new transfer_fn_wrapper2<F,G>(f,g))
    {}

    void apply_grad( cube<double>& dEdF, const cube<double>& F) const
    {
        ZI_ASSERT(impl_);
        impl_->apply_grad(dEdF, F);
    }

    void apply( cube<double>& F) const
    {
        ZI_ASSERT(impl_);
        impl_->apply(F);
    }

    void add_apply(double c, cube<double>& f) const
    {
        ZI_ASSERT(impl_);
        impl_->add_apply(c, f);
    }

    double operator()(double x) const
    {
        ZI_ASSERT(impl_);
        return impl_->operator()(x);
    }

    double grad(double f) const
    {
        ZI_ASSERT(impl_);
        return impl_->grad(f);
    }

};

template<class T, class... Args>
inline transfer_fn make_transfer_fn(Args&&... args)
{
    T fn(std::forward<Args>(args)...);
    return transfer_fn(fn);
}


}} // namespace zi::znn

