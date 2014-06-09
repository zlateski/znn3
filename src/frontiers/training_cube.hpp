#pragma once

#include <fstream>
#include <string>
#include <functional>
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>

#include <zi/async.hpp>

#include "../core/types.hpp"
#include "../core/cube_utils.hpp"
#include "../core/diskio.hpp"
#include "../core/waiter.hpp"

#include "utility.hpp"

namespace zi {
namespace znn {
namespace frontiers {

template<typename T, typename F>
inline cube<T> cube_cast(const cube<F>& in)
{
    cube<T> out(in.n_rows, in.n_cols, in.n_slices);
    for ( size_t i = 0; i < in.n_elem; ++i )
        out.memptr()[i] = static_cast<T>(in.memptr()[i]);
    return out;
}

template<typename T>
inline cube<T> crop(const cube<T>& c, const vec3s& s, const vec3s& l)
{
    return c.subcube(s[0],s[1],s[2],s[0]+l[0]-1,s[1]+l[1]-1,s[2]+l[2]-1);
}

struct sample
{
    cube<double> image;
    cube<double> label;
    cube<char>   mask;

    double w_pos;
    double w_neg;
};

class training_cube
{
private:
    cube<float> image;
    cube<char>  label;
    cube<char>  mask ;

    vec3s       size_       ;
    vec3s       in_sz_      ;
    vec3s       out_sz_     ;
    vec3s       half_in_sz_ ;
    vec3s       half_out_sz_;
    vec3s       margin_sz_  ;
    vec3s       set_sz_     ;

private:
    sample                  next_sample_    ;
    bool                    has_next_sample_ = false;
    std::mutex              mutex_          ;
    std::condition_variable cv_             ;

    bool                    next_sample_pos_ = false;

    void prepare_sample()
    {
        guard g(mutex_);

        ZI_ASSERT(!has_next_sample_);

        while (1)
        {
            vec3s loc = vec3s(half_in_sz_[0] + (rand() % set_sz_[0]),
                              half_in_sz_[1] + (rand() % set_sz_[1]),
                              half_in_sz_[2] + (rand() % set_sz_[2]));

            cube<char>  cmask  = crop(mask , loc - half_out_sz_, out_sz_ );

            if ( cmask.max() < 1 ) continue;

            cube<char>  clabel = crop(label, loc - half_out_sz_, out_sz_ );

            if ( next_sample_pos_ )
            {
                if ( clabel.max() < 1 ) continue;
            }
            else
            {
                if ( out_sz_ == vec3s::one )
                {
                    if ( clabel(0,0,0) > 0.5 ) continue;
                }
            }

            next_sample_pos_ = !next_sample_pos_;

            cube<float> fimage = crop(image, loc - half_in_sz_ , in_sz_ );

            // TEMP
            //cmask.fill(1);

            if ( rand() % 2 )
            {
                flip_x_dim(clabel); flip_x_dim(cmask); flip_x_dim(fimage);
            }

            if ( rand() % 2 )
            {
                flip_y_dim(clabel); flip_y_dim(cmask); flip_y_dim(fimage);
            }

            if ( rand() % 2 )
            {
                flip_z_dim(clabel); flip_z_dim(cmask); flip_z_dim(fimage);
            }

            if ( cmask.n_rows == cmask.n_cols )
            {
                if ( rand() % 2 )
                {
                    rotate_xy(clabel); rotate_xy(cmask); rotate_xy(fimage);
                }
            }

            next_sample_= { cube_cast<double>(fimage),
                            cube_cast<double>(clabel),
                            std::move(cmask),
                            w_pos,
                            w_neg };

            has_next_sample_ = true;
            cv_.notify_one();
            return;
        }
    }


public:
    double      w_pos = 0;
    double      w_neg = 0;

    size_t      n_pos = 0;
    size_t      n_neg = 0;

public:
    training_cube(const std::string& fname,
                  const vec3s& in_sz,
                  const vec3s& out_sz)
        : in_sz_(in_sz)
        , out_sz_(out_sz)
    {
        vec3s fov = in_sz - out_sz + vec3s::one;

        std::cout << "Loading: " << fname << " ... " << std::flush;

        auto sizefn = fname + ".size";
        std::ifstream sizef(sizefn.c_str());

        zi::vl::vec<int,3> s;
        io::read(sizef, s);

        std::cout << s;

        cube<double> tmp(s[0],s[1],s[2]);

        auto imagefn = fname + ".image";
        std::ifstream imagef(imagefn.c_str());
        io::read(imagef, tmp);

        image = cube_cast<float>(tmp);
        image = mirror_cube(image, fov);

        auto labelfn = fname + ".label";
        std::ifstream labelf(labelfn.c_str());
        io::read(labelf, tmp);

        label = cube_cast<char>(tmp);
        label = mirror_cube(label, fov);

        mask.resize(s[0], s[1], s[2]);

        auto maskfn = fname + ".mask";
        std::ifstream maskf(maskfn.c_str());
        io::read(maskf, mask);

        mask = mirror_cube(mask, fov);

        size_ = size(image);

        half_in_sz_  = in_sz_/vec3i(2,2,2);
        half_out_sz_ = out_sz_/vec3i(2,2,2);

        // margin consideration for even-sized input
        margin_sz_ = half_in_sz_;
        if ( in_sz_[0] % 2 == 0 ) --(margin_sz_[0]);
        if ( in_sz_[1] % 2 == 0 ) --(margin_sz_[1]);
        if ( in_sz_[2] % 2 == 0 ) --(margin_sz_[2]);

        set_sz_ = size_ - margin_sz_ - half_in_sz_;

        for ( size_t z = half_in_sz_[2]; z < half_in_sz_[2]+set_sz_[2]; ++z )
            for ( size_t y = half_in_sz_[1]; y < half_in_sz_[1]+set_sz_[1]; ++y )
                for ( size_t x = half_in_sz_[0]; x < half_in_sz_[0]+set_sz_[0]; ++x )
                {
                    if ( mask(x,y,z) )
                    {
                        if ( label(x,y,z) )
                            ++n_pos;
                        else
                            ++n_neg;
                    }
                }

        w_pos = w_neg = n_pos + n_neg;

        w_pos /= 2 * n_pos;
        w_neg /= 2 * n_neg;

        zi::async::async(&training_cube::prepare_sample, this);

        std::cout << " DONE" << std::endl;
    }

    sample get_sample()
    {
        guard g(mutex_);

        while (!has_next_sample_)
        {
            cv_.wait(g);
        }

        has_next_sample_ = false;
        zi::async::async(&training_cube::prepare_sample, this);

        return std::move(next_sample_);
    }
};

class training_cubes
{
private:
    std::vector<std::unique_ptr<training_cube>> cubes_;

// private:
//     void load_training_cube( std::unique_ptr<training_cube>& tc,
//                              std::string fname,
//                              const vec3s& in_sz,
//                              const vec3s& out_sz,
//                              waiter& w)
//     {
//         unique_ptr<training_cube> t(new training_cube(fname, in_sz, out_sz));
//         tc = std::move(t);
//         w.one_done();
//     }

public:
    training_cubes(const std::string& fname,
                   const std::vector<size_t> nums,
                   const vec3s& in_sz,
                   const vec3s& out_sz)
    {
        for ( auto a: nums )
        {
            cubes_.emplace_back(
                new training_cube(fname + std::to_string(a), in_sz, out_sz));
        }
    }

    sample get_sample()
    {
        return cubes_[rand()%cubes_.size()]->get_sample();
    }
};


}}} // namespace zi::znn::frontiers
