#pragma once

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <cstdio>
#include <zi/vl/vl.hpp>

#include "types.hpp"

namespace zi {
namespace znn {
namespace io {

template<typename T, typename Char, typename CharT>
inline T read(std::basic_istream<Char,CharT>& in)
{
    T r;
    in.read(reinterpret_cast<char*>(&r), sizeof(T));
    if ( !in )
    {
        throw std::runtime_error("Not enough bytes in the stream");
    }
    return r;
}

template<typename T, typename Char, typename CharT,
         class = typename std::enable_if<std::is_pod<T>::value>::type >
inline void read(std::basic_istream<Char,CharT>& in, T& r)
{
    in.read(reinterpret_cast<char*>(&r), sizeof(T));
    if ( !in )
    {
        throw std::runtime_error("Not enough bytes in the stream");
    }
}

template<typename T, typename Char, typename CharT,
         class = typename std::enable_if<std::is_pod<T>::value>::type >
inline void write(std::basic_ostream<Char,CharT>& out, const T& v)
{
    out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template<typename T, typename Char, typename CharT>
inline void read_n(std::basic_istream<Char,CharT>& in, T* r, std::size_t n)
{
    if ( n == 0 )
    {
        throw std::logic_error("Asked to read 0 bytes?");
    }
    in.read(reinterpret_cast<char*>(r), sizeof(T)*n);
    if ( !in )
    {
        throw std::runtime_error("Not enough bytes in the stream");
    }
}

template<typename T, typename Char, typename CharT>
inline void write_n(std::basic_ostream<Char,CharT>& out, const T* v, std::size_t n)
{
    out.write(reinterpret_cast<const char*>(v), sizeof(T)*n);
}


template<typename Char, typename CharT, typename T, std::size_t N>
void read(std::basic_istream<Char, CharT>& in, zi::vl::vec<T,N>& v)
{
    for ( std::size_t i = 0; i < N; ++i )
    {
        v[i] = read<T>(in);
    }
}

template<typename Char, typename CharT, typename T, std::size_t N>
void write(std::basic_ostream<Char, CharT>& out, const zi::vl::vec<T,N>& v)
{
    for ( std::size_t i = 0; i < N; ++i )
    {
        write(out, v[i]);
    }
}

template<typename Char, typename CharT, typename T>
void read(std::basic_istream<Char, CharT>& in, cube<T>& c)
{
    read_n(in, c.memptr(), c.n_elem);
}

template<typename Char, typename CharT, typename T>
void write(std::basic_ostream<Char, CharT>& out, const cube<T>& c)
{
    write_n(out, c.memptr(), c.n_elem);
}


}}} // namespace zi::znn::io
