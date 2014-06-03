#pragma once

#include <string>
#include <fstream>

#include "../core/cube_utils.hpp"
#include "../core/types.hpp"
#include "../core/diskio.hpp"

namespace zi {
namespace znn {
namespace frontiers {


template<typename T>
cube<T> mirror_cube( const cube<T>& c, const vec3s& fov)
{
    cube<T> r(c.n_rows+fov[0]-1, c.n_cols+fov[1]-1, c.n_slices+fov[2]-1);
    vec3s is  = size(c);
    vec3s os  = size(r);
    vec3s off = fov;
    off /= 2;

    r.subcube(off[0],off[1],off[2],
              off[0]+is[0]-1,off[1]+is[1]-1,off[2]+is[2]-1) = c;

    for ( size_t x = 0; x < off[0]; ++x )
    {
        r.subcube(off[0]-x-1,0,0,off[0]-x-1,os[1]-1,os[2]-1)
            = r.subcube(off[0]+x,0,0,off[0]+x,os[1]-1,os[2]-1);

        if ( is[0]+off[0]+x < os[0] )
            r.subcube(is[0]+off[0]+x,0,0,is[0]+off[0]+x,os[1]-1,os[2]-1)
                = r.subcube(is[0]+off[0]-x-1,0,0,
                            is[0]+off[0]-x-1,os[1]-1,os[2]-1);
    }

    for ( size_t y = 0; y < off[1]; ++y )
    {
        r.subcube(0,off[1]-y-1,0,os[0]-1,off[1]-y-1,os[2]-1)
            = r.subcube(0,off[1]+y,0,os[0]-1,off[1]+y,os[2]-1);

        if ( is[1]+off[1]+y < os[1] )
            r.subcube(0,is[1]+off[1]+y,0,os[0]-1,is[1]+off[1]+y,os[2]-1)
                = r.subcube(0,is[1]+off[1]-y-1,0,
                            os[0]-1,is[1]+off[1]-y-1,os[2]-1);
    }

    for ( size_t z = 0; z < off[2]; ++z )
    {
        r.subcube(0,0,off[2]-z-1,os[0]-1,os[1]-1,off[2]-z-1)
            = r.subcube(0,0,off[2]+z,os[0]-1,os[1]-1,off[2]+z);

        if ( is[2]+off[2]+z < os[2] )
            r.subcube(0,0,is[2]+off[2]+z,os[0]-1,os[1]-1,is[2]+off[2]+z)
                = r.subcube(0,0,is[2]+off[2]-z-1,
                            os[0]-1,os[1]-1,is[2]+off[2]-z-1);
    }


    return r;
}

template<typename T>
cube<T> load_cube( const std::string& fname )
{
    auto sizefn = fname + ".size";
    std::ifstream sizef(sizefn.c_str());

    zi::vl::vec<int,3> s;
    io::read(sizef, s);

    cube<double> ret(s[0],s[1],s[2]);

    auto imagefn = fname + ".image";
    std::ifstream imagef(imagefn.c_str());
    io::read(imagef, ret);

    return ret;
}

template<typename T>
void save_cube( const std::string& fname, const cube<T>& c )
{
    auto sizefn = fname + ".size";
    std::ofstream sizef(sizefn.c_str());

    zi::vl::vec<int,3> s(c.n_rows, c.n_cols, c.n_slices);
    io::write(sizef, s);

    auto imagefn = fname + ".image";
    std::ofstream imagef(imagefn.c_str());
    io::write(imagef, c);
}


template<typename N>
void process_whole_cube( const std::string& ifname,
                         const std::string& ofname,
                         N& net,
                         size_t cube_width = 32)
{
    auto c = load_cube<double>(ifname);
    auto r = c;

    c = mirror_cube(c, net.fov());

    vec3s os = size(r);
    vec3s fov = net.fov();

    std::vector<cube<double>> input(1);

    for ( std::size_t z = 0; z < os[2]; z += cube_width )
        for ( std::size_t y = 0; y < os[1]; y += cube_width )
            for ( std::size_t x = 0; x < os[0]; x += cube_width )
            {
                size_t zi = std::min(z + cube_width, os[2]);
                size_t yi = std::min(y + cube_width, os[1]);
                size_t xi = std::min(x + cube_width, os[0]);

                input[0] = c.subcube(x,y,z,xi+fov[0]-2,yi+fov[1]-2,zi+fov[2]-1);

                auto output = net.forward(input);

                r.subcube(x,y,z,xi-1,yi-1,zi-1) = output[0];

                std::cout << vec3s(x,y,z) << std::endl;
            }

    save_cube( ofname, r );

}



}}} // namespace zi::znn::frontiers
