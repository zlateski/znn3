#include "core/sum_of.hpp"

#include "network/advanced/input_layer.hpp"
#include "network/advanced/convolutional_layer.hpp"

#include <future>
#include <iostream>
#include <thread>
#include <sstream>
#include <fstream>
#include <vector>
#include <functional>
#include <zi/time.hpp>

//#include "core/fft.hpp"
#include "frontiers/training_cube.hpp"
#include "frontiers/square_loss.hpp"
#include "frontiers/cross_entropy_loss.hpp"
#include "frontiers/utility.hpp"

#include "pooling/pooling_filter_2.hpp"
#include "core/tube_iterator.hpp"
#include "pooling/pooling_filter.hpp"

#include "core/types.hpp"
#include "core/diskio.hpp"
#include "core/waiter.hpp"

#include "network/layered_network_data.hpp"
#include "network/simple_network.hpp"
#include "network/simple_network_two.hpp"
#include "network/parallel_network.hpp"
#include "transfer_fn/transfer_fn.hpp"

#include "convolution/sparse_convolve.hpp"
#include "convolution/constant_convolve.hpp"

//#include "core/concept.hpp"

//CONCEPT_IS_CALLABLE(push_back);

namespace arma {
thread_local arma_rng_cxx11 arma_rng_cxx11_instance;
}

using namespace zi::znn;

cube<double> images[60000];
std::vector<cube<double>> labels[60000];

sum_of<unique_cube<double>> summer;

std::atomic<int> total;

void do_one()
{
    auto x = pool<double>::get_unique(5,5,5);
    x->randu();

    if ( summer.add(x) )
    {
        std::cout << ++total << std::endl;
    }
}

void do_something1()
{
    std::cout << "do_something()" << std::endl;
}

void do_something2(int i)
{
    std::cout << "do_something(" << i << ")" << std::endl;
}


void do_something3( std::unique_ptr<int> i )
{
    std::cout << "do_something3(" << *i << ")" << std::endl;
}

int main()
{

//zi::async::set_concurrency(1);

    // total = 0;

    // summer.init(10);

    // for ( int i = 0; i < 1000; ++i )
    // {
    //     zi::async::async(do_one);
    // }

    // //sleep(1000);


    if (0)
    {
        std::ifstream netf("my_netqq");

        layered_network net1(netf); // 28
        layered_network_data nld(net1);
        parallel_network snet(nld, make_transfer_fn<sigmoid>());

        for ( int i = 56; i <= 56; ++i )
        {
            //std::string ifname = "/data/home/zlateski/uygar/test/confocal" + std::to_string(i);
            std::string ifname = "/data/home/zlateski/uygar/data_24Jan2014/confocal" + std::to_string(i);

            std::string ofname = "./test/" + std::to_string(i);

            frontiers::process_whole_cube(ifname, ofname, snet, 100, false);
        }

        return 0;
    }

    if (1)
    {

        std::vector<size_t> cells{56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72};

        frontiers::training_cubes tc
            ("/data/home/zlateski/uygar/data_24Jan2014/confocal", cells,
             vec3s(19,19,1), vec3s(11,11,1));

        frontiers::sample s = tc.get_sample();


        layered_network net1(1); // 28

        std::ifstream sn("my_net2");
        // if ( sn )
        // {
        //     net1.read(sn);
        // }
        // else
        {
            net1.add_layer(8,vec3s(5,5,1),0.0001);
            //net1.add_layer(8,vec3s(3,3,3),0.001);
            net1.add_layer(8,vec3s(5,5,1),0.0001);
            // net1.add_layer(15,vec3s(3,3,3),0.0001);
            // net1.add_layer(15,vec3s(5,5,1),0.0001);
            net1.add_layer(1,vec3s(1,1,1),0.0001);
        }

        layered_network_data nld(net1);
        parallel_network snet(nld, make_transfer_fn<hyperbolic_tangent>());

        double clerr= 0;
        double err  = 0;
        int    iter = 0;

        while (1)
        {
            frontiers::sample s = tc.get_sample();
            //s.w_pos = s.w_neg = 1;

            std::vector<cube<double>> input;
            input.push_back(s.image);


            std::vector<cube<double>> guess
                = snet.forward(input);


            std::vector<cube<double>> grad(1);

            auto x = frontiers::square_loss(s, guess[0]);


            grad[0] = std::move(std::get<3>(x));
            //grad[1] = std::move(std::get<4>(x));

            iter += std::get<0>(x);
            err  += std::get<1>(x);
            clerr+= std::get<2>(x);


            if ( iter > 500000 )
            {
                std::cout << "CL: " << (clerr/iter) << std::endl;
                std::cout << "SQ: " << (err/iter) << std::endl;
                iter = 0;
                err = 0;
                clerr = 0;
                std::ofstream sn("my_netqq");
                net1.write(sn);
            }

            snet.backward(grad);
            snet.grad_update();


            //std::cout << guess[0] << "\n";
        }



    //net1.add_layer(10,vec3s(2,2,1),0.01); // 1
    //net1.add_layer(50,vec3s(1,1,1),0.002);

    //simple_network snet(net1, make_transfer_fn<sigmoid>());



        //cube<char>;
        //delete c;

        auto a = pool<bool>::get_unique(4,4,4);
        a->randu();

        std::ofstream o("test.bool");
        io::write(o, *a);

        // auto b1 = pool<double>::get_unique(4,4,4);
        // b1->randu();

        // *b1 *= 2;

        // *a = *b1;

        // std::cout << *a;

        // (*b1)(0,0,0) = 0.2;

        // auto b2 = pool<double>::get_unique_zero(4,4,4);
        // (*b2)(0,0,0) = 0.2;

        // auto rd = convolve_flipped(*a,*b1);

        // auto afft = fftw::forward(*a);
        // auto bfft = fftw::forward(*b2);

        // pairwise_mult(*afft,*bfft);

        // auto rf = fftw::backward(*afft,vec3s(4,4,4));

        // *rf /= 64;

        // std::cout << *rd << std::endl;
        // std::cout << *rf << std::endl;


        // auto f = fftw::forward(*v);
        // auto b = fftw::backward(*f, size(*v));

        // auto s = size(*v);

        // *b /= ( s[0] * s[1] * s[2] );




        // std::cout << *v - *b << "\n";

    }
    //return 0;



    std::ifstream flabels("/data/home/zlateski/labels.raw");
    uint32_t magic = io::read<uint32_t>(flabels);
    std::cout << "Magic: " << magic << std::endl;

    uint32_t nimages = io::read<uint32_t>(flabels);
    std::cout << "Nimages: " << nimages << std::endl;

    for ( int i = 0; i < 60000; ++i )
    {
        for ( int j = 0; j < 10; ++j )
        {
            labels[i].push_back(cube<double>(1,1,1));
            labels[i][j](0,0,0) = 0;
        }

        uint8_t b = io::read<uint8_t>(flabels);
        labels[i][b](0,0,0) = 1;

        //std::cout << i << ' ' << int(b) << "\n";
    }

    std::ifstream fimages("/data/home/zlateski/images.raw");
    magic = io::read<uint32_t>(fimages);
    std::cout << "Magic: " << magic << std::endl;

    nimages = io::read<uint32_t>(fimages);
    std::cout << "Nimages: " << nimages << std::endl;

    uint32_t width = io::read<uint32_t>(fimages);
    std::cout << "Width: " << width << std::endl;

    uint32_t height = io::read<uint32_t>(fimages);
    std::cout << "Height: " << height << std::endl;

    for ( int i = 0; i < 60000; ++i )
    {
        images[i] = cube<double>(28,28,1);
        cube<uint8_t> cc(28,28,1);
        io::read(fimages,cc);

        for ( int a = 0; a < 28; ++a)
            for ( int b = 0; b < 28; ++b)
                images[i](a,b,0) = int(cc(a,b,0));

        images[i] /= 255;
    }


    layered_network net1(1); // 28
    net1.add_layer(10,vec3s(9,9,1),vec3s(1,1,1),0.01); // 28
    net1.add_layer(10,vec3s(9,9,1),0.01); // 20
    net1.add_layer(10,vec3s(9,9,1),0.01); // 12
    net1.add_layer(10,vec3s(4,4,1),0.01); // 4
    //net1.add_layer(10,vec3s(2,2,1),0.01); // 1
    //net1.add_layer(50,vec3s(1,1,1),0.002);

    //simple_network snet(net1, make_transfer_fn<sigmoid>());

    layered_network_data nld(net1);


    //simple_network_two snet(nld, make_transfer_fn<sigmoid>());

    parallel_network snet(nld, make_transfer_fn<sigmoid>());

    double sqerr = 0;
    double clerr = 0;
    int iter = 0;


    std::cout << "FOV: " << net1.fov() << "\n";

    for (;;)
    {
        std::vector<cube<double>> input;

        int sample_no = rand()%60000;
        // sample_no = (iter % 2)*2000+1231;

        input.push_back(images[sample_no]);

        zi::wall_timer wt;
        wt.restart();

        std::vector<cube<double>> guess
            = snet.forward(input);



        // std::cout << "GUESS: ";
        // for ( const auto& a: guess ) std::cout << ' ' << a(0,0,0);
        // std::cout << "\n\tLABEL: ";
        // for ( const auto& l: labels[sample_no] ) std::cout << ' ' << l(0,0,0);
        // std::cout << std::endl;


        //std::cout << size(guess[0]) << " ----" << std::endl;

        int mind = 0;
        double high = -1e20;

        for ( int i = 0; i < 10; ++i )
        {
            sqerr += (guess[i](0,0,0)-labels[sample_no][i](0,0,0))
                *(guess[i](0,0,0)-labels[sample_no][i](0,0,0));

            if ( guess[i](0,0,0) > high )
            {
                high = guess[i](0,0,0);
                mind = i;
            }

            guess[i] -= labels[sample_no][i];
            guess[i] *= 2;
        }

        //guess[mind] *= 100;

        //std::cout << "FWD IN: " << wt.elapsed<double>() << std::endl;

        //std::cout << guess[0](0,0,0) << ' ' << label << std::endl;

        double cl = (labels[sample_no][mind](0,0,0) > 0.5) ? 0 : 1;
        clerr += cl;

        ++iter;


        if ( iter % 1000 == 0 )
        {
            std::cout << "SQERR: " << (sqerr/iter) << std::endl;
            std::cout << "CLERR: " << (clerr/iter) << std::endl;
            std::cout << "CLERR: " << (clerr) << std::endl;
            sqerr = 0;
            clerr = 0;
            iter = 0;
        }

        // std::cout << input[0] << std::endl;
        // std::cout << "LABEL: " << label << std::endl;

        // std::cout << "GUESS: " << guess[0](0,0,0) << "\n"
        //           << "LABEL: " << label << "\n";

        // std::cout << "ERRRRR: " << guess[0](0,0,0) << "\n";

        wt.reset();


        snet.backward(guess);

        //std::cout << "BWD IN: " << wt.elapsed<double>() << std::endl;

        wt.reset();

        snet.grad_update();

        //std::cout << "GRAD IN: " << wt.elapsed<double>() << std::endl;


    }

}
