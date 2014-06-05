#pragma once

#include "types.hpp"

namespace zi {
namespace znn {

template< class U >
class sum_of: public U
{
private:
    size_t     total_ = 0;
    size_t     sofar_ = 0;
    std::mutex mutex_;

public:
    sum_of()
        : U()
    {}

    explicit sum_of(size_t n)
        : U()
        , total_(n)
        , sofar_(0)
    {}

    sum_of(sum_of&& oth)
        : U(std::move(oth))
        , total_(oth.total_)
        , sofar_(oth.sofar_)
        , mutex_(std::move(oth.mutex_))
    {}

    sum_of& operator=(sum_of&& oth)
    {
        U::operator=(std::move(oth));
        total_ = oth.total_;
        sofar_ = oth.sofar_;
        mutex_ = std::move(oth.mutex_);
        return *this;
    }

    void init(size_t n)
    {
        guard g(mutex_);
        total_ = n;
        sofar_ = 0;
        U::reset();
    }

    bool add(U& v)
    {
        while (1)
        {
            U old;
            {
                guard g(mutex_);

                if ( sofar_ == 0 )
                {
                    ++sofar_;
                    U::operator=(std::move(v));
                    if ( sofar_ == total_ )
                    {
                        sofar_ = 0;
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    if ( U::operator bool() )
                    {
                        old = std::move(*this);
                    }
                    else
                    {
                        ++sofar_;
                        U::operator=(std::move(v));
                        if ( sofar_ == total_ )
                        {
                            sofar_ = 0;
                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    }
                }
            }
            *v += *old;
        }
    }

}; // class sum_of

}} // namespace zi::znn
