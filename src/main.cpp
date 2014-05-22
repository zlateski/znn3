#include <iostream>
#include <thread>
#include <sstream>
#include <fstream>
#include <vector>
#include <functional>
#include <zi/time.hpp>

#include "core/types.hpp"
#include "core/diskio.hpp"
#include "core/waiter.hpp"

#include "network/layered_network_data.hpp"
#include "network/simple_network.hpp"
#include "network/simple_network_two.hpp"
#include "network/parallel_network.hpp"
#include "transfer_fn/transfer_fn.hpp"

//#include "core/concept.hpp"

//CONCEPT_IS_CALLABLE(push_back);

namespace arma {
thread_local arma_rng_cxx11 arma_rng_cxx11_instance;
}



struct perax
{
    perax()
    {
        std::cout << "PERA!" << std::endl;
    };

    void print(const std::string& what)
    {
        std::cout << what << std::endl;
    }
};

thread_local perax px;

void nada()
{
    px.print("nada");
}

using namespace zi::znn;


cube<double> images[60000];
std::vector<cube<double>> labels[60000];

void deleter(int* x)
{
    std::cout << "DELETE: " << *x << "\n";
    delete x;
}

void report(waiter& w)
{
    std::cout << "Reporting" << std::endl;
    w.one_done();
}

int main()
{

    zi::async::set_concurrency(16);

    //std::unique_ptr<int,decltype(dtr)>(new int(3), dtr);

    unique_cube<double> uc;

    if ( uc )
    {
        std::cout << "UCCCCCCC\n";
    }

    unique_cube<double> wc = pool<double>::get_unique(1,2,3);

    uc = std::move(wc);

    if ( uc )
    {
        std::cout << "UCCCCCCC\n";
    }

    if ( !wc )
    {
        std::cout << "!WCCCC\n";
    }

    std::cout << std::endl;

    {
        auto la = pool<double>::get_unique(1,2,3);
        la->fill(5);
    }

    std::cout << (*(pool<double>::get_unique(1,2,3))) << "\n";

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


    px.print("init");
    std::stringstream ss;
    int x = 3;
    int y = 0;

    io::write(ss, x);
    y = io::read<int>(ss);

    std::cout << y << "\n";

    x = 4;
    io::write(ss, x);
    y = io::read<int>(ss);
    std::cout << y << "\n";

    vec3s v0(1,2,3);
    vec3s v1;

    io::write(ss, v0);


    cube<double> c1(3,3,3); c1.fill(1);
    cube<double> c2(3,3,3);

    io::write(ss,c1);

    io::read(ss, v1);

    io::read(ss, c2);


    std::cout << v1 << "\n";
    std::cout << c2 << "\n";



    network_layer n1(3,3,vec3s(3,3,3),vec3s(1,1,1),3);
    std::stringstream s;
    n1.write(s);
    network_layer n2(s);
    network_layer n3(3,3,vec3s(3,3,3),vec3s(1,1,1),3);

    std::cout << "EQ? " << (n1==n2) << std::endl;
    std::cout << "EQ? " << (n1==n3) << std::endl;

    network_layer n4;
    std::cout << "EX? " << static_cast<bool>(n4) << std::endl;

    n4 = std::move(n2);
    std::cout << "EQ? " << (n1==n4) << std::endl;

    std::cout << "EX? " << static_cast<bool>(n2) << std::endl;

    layered_network net1(1);
    net1.add_layer(10,vec3s(9,9,1),0.01);
    net1.add_layer(10,vec3s(9,9,1),0.01);
    net1.add_layer(10,vec3s(9,9,1),0.01);
    net1.add_layer(10,vec3s(4,4,1),0.002);

    //simple_network snet(net1, make_transfer_fn<hyperbolic_tangent>());

    layered_network_data nld(net1);


    simple_network_two asnet(nld, make_transfer_fn<sigmoid>());

    parallel_network snet(nld, make_transfer_fn<sigmoid>());

    double sqerr = 0;
    double clerr = 0;
    int iter = 0;

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

        //std::cout << "FWD IN: " << wt.elapsed<double>() << std::endl;


        // std::cout << "GUESS: ";
        // for ( const auto& a: guess ) std::cout << ' ' << a(0,0,0);
        // std::cout << "\nLABEL: ";
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

    }

}
