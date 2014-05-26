#pragma once

#include <zi/utility/singleton.hpp>
#include <mutex>
#include <cstddef>
#include <list>
#include <functional>
#include <iostream>
#include <set>

#include "types.hpp"


namespace zi {
namespace znn {

// GCC doesn't know how to convert unique_ptrs with different deleters
// yet, so we have to do some mambo jumbo around it which actually
// makes us define a new type - unique_cube that we'll use when dealing
// with cashed unique cubes.

// The declaration of the unique_cashed_cube_deleter, which will be
// implemented after we have the singleton class

template<typename T>
struct unique_cashed_cube_deleter;

// Our new favorite type :)

template<typename T>
using unique_cube =
    std::unique_ptr<cube<T>,
                    unique_cashed_cube_deleter<T>>;

// The meat

template<typename T>
class single_size_cube_pool
{
private:
    vec3s                      size_;
    std::list<cube<T>*>        list_;
    std::mutex                 m_   ;

public:
    void clear()
    {
        std::lock_guard<std::mutex> g(m_);
        for ( auto& v: list_ )
        {
            delete v;
        }
        list_.clear();
    }

public:
    void return_cube( cube<T>* c )
    {
        std::lock_guard<std::mutex> g(m_);
        list_.push_back(c);
    }

public:
    single_size_cube_pool( const vec3s& s )
        : size_{s}
        , list_{}
        , m_{}
    {}

    ~single_size_cube_pool()
    {
        clear();
    }

    cube_ptr<T> get()
    {
        cube<T>* r = nullptr;
        {
            std::lock_guard<std::mutex> g(m_);
            if ( list_.size() > 0 )
            {
                r = list_.back();
                list_.pop_back();
            }
        }

        if ( !r )
        {
            r = new cube<T>(size_[0],size_[1],size_[2]);
        }

        return cube_ptr<T>(r,
                           std::bind(&single_size_cube_pool::return_cube,
                                     this, std::placeholders::_1));

    }

    unique_cube<T> get_unique()
    {
        cube<T>* r = nullptr;
        {
            std::lock_guard<std::mutex> g(m_);
            if ( list_.size() > 0 )
            {
                r = list_.back();
                list_.pop_back();
            }
        }

        if ( !r )
        {
            r = new cube<T>(size_[0],size_[1],size_[2]);
        }

        return unique_cube<T>(r);
    }

};

template< typename T >
class single_type_cube_pool
{
private:
    std::mutex                                   m_;
    std::map<vec3s, single_size_cube_pool<T>*>   pools_;

    single_size_cube_pool<T>* get_pool( const vec3s s )
    {
        std::lock_guard<std::mutex> g(m_);
        if ( pools_.count(s) )
        {
            return pools_[s];
        }
        else
        {
            single_size_cube_pool<T>* r = new single_size_cube_pool<T>{s};
            pools_[s] = r;
            return r;
        }
    }

public:
    cube_ptr<T> get( const vec3s& s )
    {
        return get_pool(s)->get();
    }

    unique_cube<T> get_unique( const vec3s& s )
    {
        return get_pool(s)->get_unique();
    }

    void return_cube( cube<T>* c )
    {
        get_pool(vec3s(c->n_rows, c->n_cols, c->n_slices))->return_cube(c);
    }

}; // single_type_cube_pool


template< typename T >
struct pool
{
private:
    static single_type_cube_pool<T>& instance;

public:
    static cube_ptr<T> get( const vec3s& s )
    {
        return instance.get(s);
    }

    static cube_ptr<T> get( size_t x, size_t y, size_t z )
    {
        return instance.get( vec3s(x,y,z) );
    }

    static unique_cube<T> get_unique( const vec3s& s )
    {
        return instance.get_unique(s);
    }

    static unique_cube<T> get_unique_zero( const vec3s& s )
    {
        unique_cube<T> ret = get_unique(s);
        ret->fill(0);
        return ret;
    }

    static unique_cube<T> get_unique( size_t x, size_t y, size_t z )
    {
        return instance.get_unique( vec3s(x,y,z) );
    }

    static unique_cube<T> get_unique_zero( size_t x, size_t y, size_t z )
    {
        return get_unique_zero(vec3s(x,y,z));
    }

    static unique_cube<T> get_unique_copy( const cube<T>& c )
    {
        unique_cube<T> ret = get_unique( c.n_rows, c.n_cols, c.n_slices );
        *ret = c;
        return ret;
    }

    // Actually private - don't use

    static void return_cube( cube<T>* c )
    {
        instance.return_cube(c);
    }
};

template< typename T >
single_type_cube_pool<T>& pool<T>::instance =
    zi::singleton<single_type_cube_pool<T>>::instance();


template<typename T>
struct unique_cashed_cube_deleter
{
    void operator()(cube<T>* c) const
    {
        pool<T>::return_cube(c);
    }
};


}} // namespace zi::znn
