#pragma once

#include <iterator>
#include <cstddef>
#include <algorithm>

#include "types.hpp"


namespace zi {
namespace znn {

namespace detail {
struct x_direction_tag {};
struct y_direction_tag {};
struct z_direction_tag {};
} // namespace detail

namespace {
detail::x_direction_tag x_direction;
detail::y_direction_tag y_direction;
detail::z_direction_tag z_direction;

inline void use_this_shit()
{
    static_cast<void>(x_direction);
    static_cast<void>(y_direction);
    static_cast<void>(z_direction);
}

} // anonymous namespace

template <typename T>
class tube_iterator
{
public:
    typedef std::ptrdiff_t                  difference_type;
    typedef T                               value_type;
    typedef T&                              reference;
    typedef T*                              pointer;
    typedef std::random_access_iterator_tag iterator_category;
    typedef std::size_t                     size_type;

private:
    T*              current_;
    difference_type delta_  ;

public:
    tube_iterator(const tube_iterator& other)
        : current_(other.current_)
        , delta_(other.delta_)
    {}

    tube_iterator(T* current, difference_type delta)
        : current_(current)
        , delta_(delta)
    {}

    tube_iterator(cube<T>& cube, size_t x, size_t y, size_t z,
                  detail::x_direction_tag, size_t sparse = 1)
        : current_(&cube(x,y,z))
        , delta_(sparse)
    {}

    tube_iterator(cube<T>& cube, size_t x, size_t y, size_t z,
                  detail::y_direction_tag, size_t sparse = 1)
        : current_(&cube(x,y,z))
        , delta_(cube.n_rows * sparse)
    {}

    tube_iterator(cube<T>& cube, size_t x, size_t y, size_t z,
                  detail::z_direction_tag, size_t sparse = 1)
        : current_(&cube(x,y,z))
        , delta_(cube.n_rows * cube.n_cols * sparse)
    {}

    ~tube_iterator()
    {}

    tube_iterator& operator=(const tube_iterator& other)
    {
        current_ = other.current_;
        delta_   = other.delta_;
    }

    tube_iterator& operator++()
    {
        current_ += delta_;
        return *this;
    }

    reference operator*()
    {
        return *current_;
    }

    friend void swap(tube_iterator& lhs, tube_iterator& rhs)
    {
        std::swap(lhs.current_, rhs.current_);
        std::swap(lhs.delta_, rhs.delta_);
    }

    tube_iterator operator++(int)
    {
        current_ += delta_;
        return tube_iterator(current_-delta_,delta_);
    }

    pointer operator->() const
    {
        return current_;
    }

    friend bool operator==(const tube_iterator& l, const tube_iterator& r)
    {
        return (l.current_ == r.current_) && (l.delta_ == r.delta_);
    }

    friend bool operator!=(const tube_iterator& l, const tube_iterator& r)
    {
        return (l.current_ != r.current_) || (l.delta_ != r.delta_);
    }

    tube_iterator& operator--()
    {
        current_ -= delta_;
        return *this;
    }

    tube_iterator operator--(int)
    {
        current_ -= delta_;
        return tube_iterator(current_ + delta_, delta_);
    }

    friend bool operator<(const tube_iterator& l, const tube_iterator& r)
    {
        return l.current_ < r.current_;
    }

    friend bool operator>(const tube_iterator& l, const tube_iterator& r)
    {
        return (r<l);
    }

    friend bool operator<=(const tube_iterator& l, const tube_iterator& r)
    {
        return !(l>r);
    }

    friend bool operator>=(const tube_iterator& l, const tube_iterator& r)
    {
        return !(l<r);
    }

    tube_iterator& operator+=(size_type n)
    {
        current_ += n*delta_;
        return *this;
    }

    friend tube_iterator operator+(const tube_iterator& l, size_type n)
    {
        return tube_iterator(l.current_ + n*l.delta_, l.delta_);
    }

    friend tube_iterator operator+(size_type n, const tube_iterator& l)
    {
        return tube_iterator(l.current_ + n*l.delta_, l.delta_);
    }

    tube_iterator& operator-=(size_type n)
    {
        current_ -= n*delta_;
        return *this;
    }

    friend tube_iterator operator-(const tube_iterator& l, size_type n)
    {
        return tube_iterator(l.current_ - n*l.delta_, l.delta_);
    }

    friend difference_type operator-(tube_iterator l, tube_iterator r)
    {
        ZI_ASSERT(l.delta_==r.delta_);
        difference_type del = r.current_ - l.current_;
        return del / l.delta_;
    }

    reference operator[](size_type n) const
    {
        return current_[n*delta_];
    }
};


template< typename T, typename D >
tube_iterator<T> tube_begin(cube<T>& c, size_t x, size_t y, size_t z,
                            D tag, size_t sparse = 1)
{
    return tube_iterator<T>(c,x,y,z,tag,sparse);
}

template< typename T >
tube_iterator<T> tube_end(cube<T>& c, size_t x, size_t y, size_t z,
                          detail::x_direction_tag tag, size_t sparse = 1)
{
    return tube_iterator<T>(c,x,y,z,tag,sparse) + c.n_rows;
}

template< typename T >
tube_iterator<T> tube_end(cube<T>& c, size_t x, size_t y, size_t z,
                          detail::y_direction_tag tag, size_t sparse = 1)
{
    return tube_iterator<T>(c,x,y,z,tag,sparse) + c.n_cols;
}

template< typename T >
tube_iterator<T> tube_end(cube<T>& c, size_t x, size_t y, size_t z,
                          detail::z_direction_tag tag, size_t sparse = 1)
{
    return tube_iterator<T>(c,x,y,z,tag,sparse) + c.n_slices;
}



}} // namespace zi::znn
