#pragma once

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <cstdio>
#include <vector>
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
inline void write_n(std::basic_ostream<Char,CharT>& out, const T* v,
                    std::size_t n)
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

template<typename Char, typename CharT>
void read(std::basic_istream<Char, CharT>& in, std::string& s)
{
    size_t size;
    read(in, size);
    s.resize(size);
    read_n(in, const_cast<char*>(s.data()), size);
}

template<typename Char, typename CharT>
void write(std::basic_ostream<Char, CharT>& out, const std::string& s)
{
    size_t size = s.size();
    write(out, size);
    write_n(out, s.data(), size);
}

template<typename Char, typename CharT, typename T>
void read(std::basic_istream<Char, CharT>& in, std::vector<T>& v)
{
    size_t size;
    read(in, size);
    v.resize(size);
    read_n(in, v.data(), size);
}

template<typename Char, typename CharT, typename T>
void write(std::basic_ostream<Char, CharT>& out, const std::vector<T>& v)
{
     size_t size = v.size();
     write(out, size);
     write_n(out, v.data(), size);
}


namespace detail
{

struct istream_wrapper_type_erasure
{
    virtual ~istream_wrapper_type_erasure() {};
    virtual void read(char*, size_t) = 0;
    virtual operator bool() const = 0;
};


struct ostream_wrapper_type_erasure
{
    virtual ~ostream_wrapper_type_erasure() {};
    virtual void write(const char*, size_t) = 0;
    virtual operator bool() const = 0;
};

template< class IStream >
struct istream_wrapper_type_erasure_of_stream
    : istream_wrapper_type_erasure
{
private:
    IStream& stream_;

public:
    istream_wrapper_type_erasure_of_stream( IStream& s )
        : stream_(s)
    {}

    void read(char* c, size_t n) override
    {
        stream_.read(c, n);
    }

    operator bool() const override
    {
        return stream_;
    }
};

template< class OStream >
struct ostream_wrapper_type_erasure_of_stream
    : ostream_wrapper_type_erasure
{
private:
    OStream& stream_;

public:
    ostream_wrapper_type_erasure_of_stream( OStream& s )
        : stream_(s)
    {}

    void write(const char* c, size_t n) override
    {
        stream_.write(c, n);
    }

    operator bool() const override
    {
        return stream_;
    }
};

} // namespace detail

struct istream
{
private:
    std::unique_ptr<detail::istream_wrapper_type_erasure> stream_;

public:
    template< class T >
    explicit istream(T& s)
        : stream_(new detail::istream_wrapper_type_erasure_of_stream<T>(s))
    {}

    void read(char* c, size_t n)
    {
        if ( n == 0 )
        {
            throw std::logic_error("Asked to read 0 bytes?");
        }

        stream_->read(c, n);

        if ( !(*stream_) )
        {
            throw std::runtime_error("Not enough bytes in the stream");
        }
    }

    template< class T >
    void read_n(T* r, size_t n)
    {
        read(reinterpret_cast<char*>(r), sizeof(T)*n);
    }

    operator bool() const
    {
        return static_cast<bool>(*stream_);
    }

    template< class T,
              class =
              typename std::enable_if<std::is_arithmetic<T>::value>::type>
    istream& operator>>( T& v )
    {
        read(reinterpret_cast<char*>(&v), sizeof(T));
        return *this;
    }

    template< class T,
              class =
              typename std::enable_if<!std::is_arithmetic<T>::value>::type,
              class =
              typename std::enable_if<std::is_pod<T>::value>::type>
    istream& operator>>( T& v )
    {
        read(reinterpret_cast<char*>(&v), sizeof(T));
        return *this;
    }

    istream& operator>>( std::string& s )
    {
        size_t sz;
        *this >> sz;
        s.resize(sz);
        read(const_cast<char*>(s.data()), sz);
        return *this;
    }

    template< typename T >
    istream& operator>>( std::vector<T>& v )
    {
        size_t sz;
        *this >> sz;
        v.resize(sz);
        read_n(v.data(), sz);
        return *this;
    }

    template<typename T, std::size_t N>
    istream& operator>>( zi::vl::vec<T,N>& v )
    {
        for ( size_t i = 0; i < N; ++i )
        {
            *this >> v[i];
        }
        return *this;
    }

    template<typename T>
    istream& operator>>( cube<T>& c )
    {
        read_n(c.memptr(), c.n_elem);
        return *this;
    }


}; // struct istream


struct ostream
{
private:
    std::unique_ptr<detail::ostream_wrapper_type_erasure> stream_;

public:
    template< class T >
    explicit ostream(T& s)
        : stream_(new detail::ostream_wrapper_type_erasure_of_stream<T>(s))
    {}

    void write(const char* c, size_t n)
    {
        if ( n == 0 )
        {
            throw std::logic_error("Asked to write 0 bytes?");
        }

        stream_->write(c, n);
    }

    template< class T >
    void write_n(const T* r, size_t n)
    {
        write(reinterpret_cast<const char*>(r), sizeof(T)*n);
    }

    operator bool() const
    {
        return static_cast<bool>(*stream_);
    }

    template< class T,
              class =
              typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ostream& operator<<( const T& v )
    {
        write(reinterpret_cast<const char*>(&v), sizeof(T));
        return *this;
    }

    template< class T,
              class =
              typename std::enable_if<!std::is_arithmetic<T>::value>::type,
              class =
              typename std::enable_if<std::is_pod<T>::value>::type>
    ostream& operator<<( const T& v )
    {
        write(reinterpret_cast<const char*>(&v), sizeof(T));
        return *this;
    }

    ostream& operator<<( const std::string& s )
    {
        size_t sz = s.size();
        *this << sz;
        write(s.data(), sz);
        return *this;
    }

    template< typename T >
    ostream& operator<<( const std::vector<T>& v )
    {
        size_t sz = v.size();
        *this << sz;
        write_n(v.data(), sz);
        return *this;
    }

    template<typename T, std::size_t N>
    ostream& operator<<( const zi::vl::vec<T,N>& v )
    {
        for ( size_t i = 0; i < N; ++i )
        {
            *this << v[i];
        }
        return *this;
    }

    template<typename T>
    ostream& operator<<( const cube<T>& c )
    {
        write_n(c.memptr(), c.n_elem);
        return *this;
    }

}; // struct ostream

struct iostream
    : istream, ostream
{
public:
    template< class T >
    explicit iostream(T& s)
        : istream(s)
        , ostream(s)
    {}

    operator bool() const
    {
        return istream::operator bool();
    }

}; // struct iostream


}}} // namespace zi::znn::io
