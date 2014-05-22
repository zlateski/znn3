#pragma once

#include <condition_variable>

#include "types.hpp"

namespace zi {
namespace znn {

class waiter
{
private:
    std::size_t             remaining_;
    std::mutex              mutex_;
    std::condition_variable cv_;

public:
    waiter()
        : remaining_(0)
    {}

    waiter(std::size_t how_many)
        : remaining_(how_many)
    {}

    void one_done()
    {
        guard g(mutex_);
        --remaining_;
        if ( remaining_ == 0 )
        {
            cv_.notify_all();
        }
    }

    void wait()
    {
        guard g(mutex_);
        while ( remaining_ )
        {
            cv_.wait(g);
        }
    }

    void set(std::size_t how_many)
    {
        guard g(mutex_);
        remaining_ = how_many;
        if ( remaining_ == 0 )
        {
            cv_.notify_all();
        }
    }

}; // class waiter;


}} // namespace zi::znn
