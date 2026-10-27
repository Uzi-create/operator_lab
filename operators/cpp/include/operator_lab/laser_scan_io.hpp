#pragma once
#include "laser_grid.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace operator_lab::navigation {
struct LaserScanRecord {
    std::vector<float> ranges;
    double angle_min=0,angle_increment=0,range_min=0,range_max=0,first_stamp_s=0,time_increment_s=0;
    LaserScanView view()const{return {ranges.data(),ranges.size(),angle_min,angle_increment,range_min,range_max,first_stamp_s,time_increment_s};}
};
// Minimal lossless text interchange, preserving NaN/Inf as unknown/no-return
// evidence. No ROS, hardware driver, epoch conversion or motion data is implicit.
inline void write_laser_scan(const std::filesystem::path& path,const LaserScanView& scan){
    last_ray_stamp(scan);
    std::ofstream file(path);
    file<<"OPERATOR_LASER_V1\n"<<scan.count<<' '<<std::setprecision(17)<<scan.angle_min<<' '<<scan.angle_increment<<' '
        <<scan.range_min<<' '<<scan.range_max<<' '<<scan.first_ray_stamp_s<<' '<<scan.time_increment_s<<'\n';
    file<<std::setprecision(std::numeric_limits<float>::max_digits10);
    for(std::size_t i=0;i<scan.count;++i){
        const double value=scan.ranges[i];
        if(std::isnan(value))file<<"nan";else if(std::isinf(value))file<<(value>0?"inf":"-inf");else file<<value;
        file<<(i+1==scan.count?'\n':' ');
    }
    if(!file)throw std::runtime_error("cannot write laser scan");
}
inline LaserScanRecord read_laser_scan(const std::filesystem::path& path){
    std::ifstream file(path);std::string magic;std::size_t count=0;LaserScanRecord out;
    if(!(file>>magic)||magic!="OPERATOR_LASER_V1"||
       !(file>>count>>out.angle_min>>out.angle_increment>>out.range_min>>out.range_max>>out.first_stamp_s>>out.time_increment_s)||
       count<2||count>8192)throw std::runtime_error("invalid laser scan header");
    out.ranges.reserve(count);
    for(std::size_t i=0;i<count;++i){
        std::string token;if(!(file>>token))throw std::runtime_error("truncated laser scan");
        std::size_t consumed=0;const float value=std::stof(token,&consumed);
        if(consumed!=token.size())throw std::runtime_error("invalid laser sample");
        out.ranges.push_back(value);
    }
    file>>std::ws;if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("extra laser samples");
    last_ray_stamp(out.view());return out;
}
} // namespace operator_lab::navigation
