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
#include "frontiers/reporter.hpp"

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

int main()
{

    //zi::async::set_concurrency(5);

    if (1)
    {
        std::ifstream netf("frontiers_sigmoid_2_hidden_layers_data_09Jun");

        layered_network net1(netf); // 28
        layered_network_data nld(net1);
        parallel_network snet(nld, make_transfer_fn<sigmoid>());

        for ( int i = 12; i <= 12; ++i )
        {
            std::string ifname = "/data/home/zlateski/uygar/test/confocal" + std::to_string(i);
            //std::string ifname = "/data/home/zlateski/uygar/data_09Jun2014/confocal" + std::to_string(i);

            std::string ofname = "./test/" + std::to_string(i);

            frontiers::process_whole_cube(ifname, ofname, snet, 100, false);
        }

        return 0;
    }

    if (1)
    {


        layered_network net1(1); // 28

        std::ifstream sn("frontiers_sigmoid_2_hidden_layers_data_09Jun");
        if ( sn )
        {
            net1.read(sn);
            net1.pop_layer();
            net1.add_layer(16,vec3s(3,3,3),0.1);
            net1.add_layer(1,vec3s(1,1,1),0.01);
        }
        else
        {
            net1.add_layer(16,vec3s(5,5,1),0.1);
            net1.add_layer(16,vec3s(5,5,1),0.1);
            net1.add_layer(1,vec3s(1,1,1),0.01);
        }

        std::cout << net1.fov() << std::endl;

        std::vector<size_t> cells{56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72};

        vec3s output_size(1,1,1);

        frontiers::training_cubes tc
            ("/data/home/zlateski/uygar/data_09Jun2014/confocal", cells,
             net1.fov() + output_size - vec3s::one, output_size);

        layered_network_data nld(net1);
        parallel_network snet(nld, make_transfer_fn<sigmoid>());


        frontiers::reporter reporter
            ("frontiers_sigmoid_3_hidden_layers_data_09Jun.report", 10000);
        reporter.clear();

        size_t niter = reporter.total_iterations();

        while (niter < 10000000 )
        {
            frontiers::sample s = tc.get_sample();

            std::vector<cube<double>> input;
            input.push_back(s.image);

            std::vector<cube<double>> guess = snet.forward(input);

            std::vector<cube<double>> grad(1);

            auto x = frontiers::square_loss(s, guess[0]);

            grad[0] = std::move(std::get<3>(x));

            if ( reporter.report(std::get<2>(x), std::get<1>(x),
                                 std::get<0>(x)) )
            {
                std::ofstream sn("frontiers_sigmoid_3_hidden_layers_data_09Jun");
                net1.write(sn);
            }

            snet.backward(grad);
            snet.grad_update();

            niter = reporter.total_iterations();
        }

        {
            std::ofstream sn("frontiers_sigmoid_3_hidden_layers_data_09Jun");
            net1.write(sn);
            reporter.force_save();
        }


        return 0;
    }
}
