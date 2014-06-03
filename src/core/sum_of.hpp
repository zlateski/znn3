#pragma once

#include "types.hpp"

namespace zi {
namespace znn {

template< class U >
class sum_of
{
private:
    size_t     total_ = 0;
    size_t     sofar_ = 0;
    std::mutex mutex_;
    U          sum_  ;

public:
    sum_of()
    {}

    explicit sum_of(size_t n)
        : total_(n)
        , sofar_(0)
    {}

    void init(size_t n)
    {
        guard g(mutex_);
        total_ = n;
        sofar_ = 0;
        sum_.reset();
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
                    sum_ = std::move(v);
                    return sofar_ == total_;
                }
                else
                {
                    if ( sum_ )
                    {
                        old = std::move(sum_);
                    }
                    else
                    {
                        ++sofar_;
                        sum_ = std::move(v);
                        return sofar_ == total_;
                    }
                }
            }
            *v += *old;
        }
    }

    U reset()
    {
        sofar_ = 0;
        return std::move(sum_);
    }

}; // class sum_of

}} // namespace zi::znn
