#pragma once

#include "../core/types.hpp"
#include "../core/diskio.hpp"

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include <zi/time.hpp>

namespace zi {
namespace znn {
namespace frontiers {

class reporter
{
private:
    std::string fname_;
    size_t      freq_ ;

    std::vector<double> clerr_;
    std::vector<double> error_;
    std::vector<size_t> niter_;

    double cur_clerr_ = 0;
    double cur_error_ = 0;
    size_t cur_niter_ = 0;
    size_t tot_niter_ = 0;

    zi::wall_timer timer_;

private:
    void update()
    {
        if ( cur_niter_ > 0 )
        {
            clerr_.push_back(cur_clerr_/cur_niter_);
            error_.push_back(cur_error_/cur_niter_);

            tot_niter_ += cur_niter_;
            niter_.push_back(tot_niter_);

            double elapsed = timer_.elapsed<double>();
            timer_.reset();

            double itps = cur_niter_;
            itps /= elapsed;

            cur_error_ = cur_clerr_ = 0;
            cur_niter_ = 0;

            std::cout << "Iter: " << niter_.back()
                      << " " << itps << " iteration/s"
                      << "\n\tClassification Error: " << clerr_.back()
                      << "\n\tTraining Error:       " << error_.back()
                      << std::endl;
        }
    }

private:
    void save()
    {
        std::ofstream ofs(fname_.c_str());
        io::ostream   os(ofs);

        os << freq_ << clerr_ << error_ << niter_
           << cur_clerr_ << cur_error_ << cur_niter_ << tot_niter_;
    }

    void try_load()
    {
        std::ifstream ifs(fname_.c_str());

        if ( !ifs ) return;

        io::istream   is(ifs);

        is >> freq_ >> clerr_ >> error_ >> niter_
           >> cur_clerr_ >> cur_error_ >> cur_niter_ >> tot_niter_;
    }

public:
    reporter(const std::string fname, size_t freq = 1)
        : fname_(fname)
        , freq_(freq)
    {
        try_load();
        if ( freq != 1 ) freq_ = freq;
        timer_.reset();
    }

    bool report( double clerr, double error, size_t iter )
    {
        cur_clerr_ += clerr;
        cur_error_ += error;
        cur_niter_ += iter;

        if ( cur_niter_ >= freq_ )
        {
            update();
            save();
            return true;
        }

        return false;
    }

    void force_save()
    {
        update();
        save();
    }

    size_t total_iterations() const
    {
        return tot_niter_ + cur_niter_;
    }

    void clear()
    {
        cur_error_ = cur_clerr_ = 0;
        cur_niter_ = tot_niter_ = 0;
        clerr_.clear();
        error_.clear();
        niter_.clear();
        timer_.reset();
    }

}; // class reporter

}}} // namespace zi::znn::frontiers
