#pragma once

#include <utility>

namespace zi {
namespace znn {

template<typename T>
class carrier
{
private:
    T val;

public:
    explicit carrier(T& v)
        : val(std::move(v))
    {}

    carrier(T&&) = delete;

    operator T&() const
    {
        return const_cast<T&>(val);
    }

    T& get() const
    {
        return const_cast<T&>(val);
    }

}; // class carrier

template< typename T >
inline carrier<T> carry(T& val)
{
    return carrier<T>(val);
}

template< typename T >
inline carrier<T> carry(carrier<T> val)
{
    return carry(val.get());
}

template <class T>
void carry(const T&&) = delete;

}} // namespace zi::znn
