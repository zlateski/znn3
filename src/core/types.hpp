#pragma once

#include <armadillo>
#include <complex>
#include <cstdint>
#include <mutex>
#include <memory>
#include <zi/assert.hpp>

#include <zi/vl/vl.hpp>

namespace zi {
namespace znn {

typedef std::complex<double>       complex;
typedef std::complex<float>        fcomplex;
typedef std::complex<long double>  long_complex;

typedef zi::vl::vec<std::size_t,3> vec3s;
typedef zi::vl::vec<int64_t,3>     vec3i;

template<typename T>
using cube = arma::Cube<T>;

template<typename T>
using const_cube = const arma::Cube<T>;

template<typename T>
using cube_ptr = std::shared_ptr<cube<T>>;

template<typename T>
using cube_unique_ptr = std::unique_ptr<cube<T>>;

typedef std::size_t size_t;

typedef std::unique_lock<std::mutex> guard;

template<typename T>
struct identity { typedef T type; };

}} // namespace zi::znn
