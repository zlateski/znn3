#pragma once

#include <fftw3.h>

#include <map>
#include <mutex>

#include "types.hpp"
#include "cube_pool.hpp"
#include "cube_utils.hpp"


namespace zi {
namespace znn {

inline vec3s fft_complex_size(const vec3s& s)
{
    auto r = s;
    r[0] /= 2;
    r[0] += 1;
    return r;
}

template< typename T >
inline vec3s fft_complex_size(const cube<T>& c)
{
    return fft_complex_size(size(c));
}

namespace detail {

class fftw_plans_impl
{
private:
    std::mutex                 m_;
    std::map<vec3s, fftw_plan> fwd_;
    std::map<vec3s, fftw_plan> bwd_;

public:
    ~fftw_plans_impl()
    {
        for ( auto& p: fwd_ ) fftw_destroy_plan(p.second);
        for ( auto& p: bwd_ ) fftw_destroy_plan(p.second);
    }

    fftw_plan get_forward( const vec3s& s )
    {
        guard g(m_);

        auto it = fwd_.find(s);
        if ( it != fwd_.end() )
        {
            return it->second;
        }

        cube<double>  in (s[0],s[1],s[2]);
        cube<complex> out(s[0]/2+1,s[1],s[2]);

        fftw_plan ret =
            fftw_plan_dft_r2c_3d( s[2], s[1], s[0],
                                  reinterpret_cast<double*>(in.memptr()),
                                  reinterpret_cast<fftw_complex*>(out.memptr()),
                                  FFTW_ESTIMATE );

        fwd_[s] = ret;
        return ret;
    }

    fftw_plan get_backward( const vec3s& s )
    {
        guard g(m_);

        auto it = bwd_.find(s);
        if ( it != bwd_.end() )
        {
            return it->second;
        }

        cube<complex> in (s[0],s[1],s[2]);
        cube<double>  out(s[0]/2+1,s[1],s[2]);

        fftw_plan ret =
            fftw_plan_dft_c2r_3d( s[2], s[1], s[0],
                                  reinterpret_cast<fftw_complex*>(in.memptr()),
                                  reinterpret_cast<double*>(out.memptr()),
                                  FFTW_ESTIMATE );

        bwd_[s] = ret;
        return ret;
    }

}; // class fftw_plans_impl


namespace {
fftw_plans_impl& plans = zi::singleton<fftw_plans_impl>::instance();
}

} // namespace detail


struct fftw
{
    static void forward( const cube<double>& in,
                         cube<complex>&      out )
    {
        ZI_ASSERT(in.n_rows/2+1==out.n_rows);
        ZI_ASSERT(in.n_cols==out.n_cols);
        ZI_ASSERT(in.n_slices==out.n_slices);

        fftw_plan plan = detail::plans.get_forward(size(in));

        fftw_execute_dft_r2c(plan,
                             const_cast<double*>(in.memptr()),
                             reinterpret_cast<fftw_complex*>(out.memptr()));
    }

    static void backward( const cube<complex>& in,
                          cube<double>&        out )
    {
        ZI_ASSERT(in.n_rows==out.n_rows/2+1);
        ZI_ASSERT(in.n_cols==out.n_cols);
        ZI_ASSERT(in.n_slices==out.n_slices);

        fftw_plan plan = detail::plans.get_backward(size(out));

        fftw_execute_dft_c2r(plan,
                             reinterpret_cast<fftw_complex*>(
                                 const_cast<complex*>(in.memptr())),
                             const_cast<double*>(out.memptr()));
    }

    static unique_cube<complex> forward( const cube<double>& in )
    {
        auto out = pool<complex>::get_unique(fft_complex_size(in));
        forward( in, *out );
        return out;
    }

    static unique_cube<complex> forward_copy( const cube<double>& in )
    {
        auto inp = pool<double>::get_unique_copy(in);
        auto out = pool<complex>::get_unique(fft_complex_size(in));
        forward( *inp, *out );
        return out;
    }

    static unique_cube<complex> forward_pad( const cube<double>& in,
                                             const vec3s& sparse,
                                             const vec3s& size )
    {
        auto inp = pool<double>::get_unique_zero(size);
        sparse_explode(in, *inp, sparse);

        auto out = pool<complex>::get_unique(fft_complex_size(size));
        forward( *inp, *out );
        return out;
    }

    static unique_cube<complex> forward_pad( const cube<double>& in,
                                             const vec3s& size )
    {
        auto inp = expand(in,size);

        auto out = pool<complex>::get_unique(fft_complex_size(size));
        forward( *inp, *out );
        return out;
    }

    static unique_cube<double> backward( const cube<complex>& in,
                                         const vec3s& s )
    {
        auto out = pool<double>::get_unique(s);
        backward( in, *out );
        return out;
    }


}; // struct fftw


}} // namespace zi::znn
