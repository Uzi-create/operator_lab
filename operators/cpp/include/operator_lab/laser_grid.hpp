#pragma once
#include "planar_geometry.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace operator_lab::navigation {
enum class Cell : std::uint8_t { unknown,free,occupied };
struct GridConfig {double xmin=-3,ymin=-3,xmax=3,ymax=3,resolution=.05;};
class LocalGrid {
public:
    explicit LocalGrid(GridConfig config={}):config_(config){
        for(double value:{config.xmin,config.ymin,config.xmax,config.ymax,config.resolution})
            if(!std::isfinite(value))throw std::invalid_argument("nonfinite grid configuration");
        if(config.resolution<.01||config.resolution>1||config.xmax<=config.xmin||config.ymax<=config.ymin)
            throw std::invalid_argument("invalid local grid extent or resolution");
        const double w=std::ceil((config.xmax-config.xmin)/config.resolution);
        const double h=std::ceil((config.ymax-config.ymin)/config.resolution);
        if(w>2048||h>2048||w*h>4000000)throw std::invalid_argument("local grid exceeds resource limit");
        width_=static_cast<std::size_t>(w);height_=static_cast<std::size_t>(h);
        cells_.assign(width_*height_,Cell::unknown);
    }
    const GridConfig& config()const{return config_;}
    std::size_t width()const{return width_;}std::size_t height()const{return height_;}
    std::optional<std::size_t> index(Point2 point)const{
        if(!finite(point)||point.x<config_.xmin||point.y<config_.ymin||point.x>=config_.xmax||point.y>=config_.ymax)return std::nullopt;
        const auto x=static_cast<std::size_t>(std::floor((point.x-config_.xmin)/config_.resolution));
        const auto y=static_cast<std::size_t>(std::floor((point.y-config_.ymin)/config_.resolution));
        if(x>=width_||y>=height_)return std::nullopt;return y*width_+x;
    }
    Point2 center(std::size_t index)const{
        if(index>=cells_.size())throw std::out_of_range("grid index");
        return {config_.xmin+(static_cast<double>(index%width_)+.5)*config_.resolution,
                config_.ymin+(static_cast<double>(index/width_)+.5)*config_.resolution};
    }
    Cell at(std::size_t index)const{return cells_.at(index);}
    Cell at(Point2 point)const{const auto i=index(point);return i?cells_[*i]:Cell::unknown;}
    void mark_free(Point2 point){if(auto i=index(point);i&&cells_[*i]!=Cell::occupied)cells_[*i]=Cell::free;}
    void mark_occupied(Point2 point){if(auto i=index(point))cells_[*i]=Cell::occupied;}
    void fill(Cell value){std::fill(cells_.begin(),cells_.end(),value);}
    std::size_t size()const{return cells_.size();}
private:
    GridConfig config_;std::size_t width_=0,height_=0;std::vector<Cell> cells_;
};
struct LaserScanView {
    const float* ranges=nullptr;std::size_t count=0;
    double angle_min=0,angle_increment=0,range_min=0,range_max=0;
    double first_ray_stamp_s=0,time_increment_s=0;
};
struct LaserConfig {
    GridConfig grid;
    double max_age_s=.2,max_ray_m=10,min_usable_fraction=.5,max_undeskewed_duration_s=.03;
    std::size_t min_usable_rays=10;
    bool positive_infinity_is_clear=false;
};
inline double last_ray_stamp(const LaserScanView& scan){
    if(!scan.ranges||scan.count<2||scan.count>8192||!std::isfinite(scan.angle_min)||
       !std::isfinite(scan.angle_increment)||scan.angle_increment==0||!std::isfinite(scan.range_min)||
       !std::isfinite(scan.range_max)||scan.range_min<0||scan.range_max<=scan.range_min||
       !std::isfinite(scan.first_ray_stamp_s)||!std::isfinite(scan.time_increment_s)||scan.time_increment_s<0)
        throw std::invalid_argument("invalid LaserScan metadata");
    const double span=std::abs(scan.angle_increment)*static_cast<double>(scan.count-1);
    const double last=scan.first_ray_stamp_s+scan.time_increment_s*static_cast<double>(scan.count-1);
    if(!std::isfinite(span)||span>2*pi+std::abs(scan.angle_increment)*1.1||!std::isfinite(last))
        throw std::invalid_argument("scan span or acquisition time invalid");
    return last;
}
struct LaserObservation {
    std::optional<LocalGrid> grid;
    bool usable=false;
    double first_stamp_s=0,last_stamp_s=0,reference_stamp_s=0;
    std::size_t usable_rays=0,obstacle_returns=0;
    std::vector<Point2> obstacle_points;
    std::string reason;
};
inline void validate(const LaserConfig& c){
    LocalGrid check(c.grid);(void)check;
    for(double value:{c.max_age_s,c.max_ray_m,c.min_usable_fraction,c.max_undeskewed_duration_s})
        if(!std::isfinite(value)||value<0)throw std::invalid_argument("invalid laser configuration");
    if(c.max_age_s<=0||c.max_ray_m<=0||c.max_ray_m>30||c.min_usable_fraction>1||!c.min_usable_rays)
        throw std::invalid_argument("invalid laser bounds");
}
// reference_from_beam_base[i] maps base coordinates at ray i into the current
// reference base. Supply odometry-derived poses; no SLAM is needed.
inline LaserObservation laser_to_grid(const LaserScanView& scan,Pose2 base_from_laser,double now_s,
    const LaserConfig& config={},const std::vector<Pose2>& reference_from_beam_base={}){
    validate(config);LaserObservation out;
    out.first_stamp_s=scan.first_ray_stamp_s;
    out.reference_stamp_s=now_s;
    if(!finite(base_from_laser)||!std::isfinite(now_s)||config.max_ray_m<=scan.range_min)
        throw std::invalid_argument("invalid laser transform, reference time or range cap");
    out.last_stamp_s=last_ray_stamp(scan);
    if(out.last_stamp_s>now_s+1e-9){out.reason="future_scan";return out;}
    if(now_s-scan.first_ray_stamp_s>config.max_age_s){out.reason="stale_scan";return out;}
    if(!reference_from_beam_base.empty()&&reference_from_beam_base.size()!=scan.count)
        throw std::invalid_argument("deskew pose count differs from scan");
    for(auto pose:reference_from_beam_base)if(!finite(pose))throw std::invalid_argument("nonfinite deskew pose");
    if(reference_from_beam_base.empty()&&out.last_stamp_s-out.first_stamp_s>config.max_undeskewed_duration_s){
        out.reason="deskew_required";return out;
    }
    LocalGrid grid(config.grid);
    // Free-space tracing samples at <=1/3 cell. Unvisited cells remain UNKNOWN.
    // Occupied endpoints always win over free rays, including later scan beams.
    for(std::size_t i=0;i<scan.count;++i){
        const double raw=scan.ranges[i];bool hit=true;double range=raw;
        if(std::isinf(raw)&&raw>0&&config.positive_infinity_is_clear){range=scan.range_max;hit=false;}
        else if(!std::isfinite(raw)||raw<scan.range_min||raw>scan.range_max||raw<=0)continue;
        ++out.usable_rays;
        if(range>config.max_ray_m){range=config.max_ray_m;hit=false;}
        const double angle=scan.angle_min+static_cast<double>(i)*scan.angle_increment;
        const Pose2 correction=reference_from_beam_base.empty()?Pose2{}:reference_from_beam_base[i];
        const auto origin=transform_point({base_from_laser.x,base_from_laser.y},correction);
        const auto endpoint=transform_point(transform_point({range*std::cos(angle),range*std::sin(angle)},base_from_laser),correction);
        if(!grid.index(origin)){out.reason="laser_origin_outside_grid";return out;}
        const auto steps=static_cast<std::size_t>(std::ceil(range/(config.grid.resolution/3)));
        for(std::size_t step=0;step<steps;++step){
            const double fraction=static_cast<double>(step)/static_cast<double>(steps);
            if(fraction*range<scan.range_min)continue; // Do not clear the sensor blind zone.
            grid.mark_free({origin.x+(endpoint.x-origin.x)*fraction,origin.y+(endpoint.y-origin.y)*fraction});
        }
        if(hit){grid.mark_occupied(endpoint);out.obstacle_points.push_back(endpoint);++out.obstacle_returns;}
        else grid.mark_free(endpoint);
    }
    if(out.usable_rays<config.min_usable_rays||static_cast<double>(out.usable_rays)/scan.count<config.min_usable_fraction){
        out.reason="insufficient_valid_rays";return out;
    }
    out.grid=std::move(grid);out.usable=true;out.reason="scan_ready";return out;
}
// Optional approximation for constant measured BODY velocity. Real odometry poses
// per ray are preferred during acceleration; repeated predictions do not refresh scan time.
inline std::vector<Pose2> constant_twist_deskew(const LaserScanView& scan,Velocity2 measured,double reference_stamp_s){
    last_ray_stamp(scan);
    if(!finite(measured)||!std::isfinite(reference_stamp_s)||scan.count>8192||
       !std::isfinite(scan.first_ray_stamp_s)||!std::isfinite(scan.time_increment_s)||scan.time_increment_s<0)
        throw std::invalid_argument("invalid scan deskew request");
    std::vector<Pose2> poses;poses.reserve(scan.count);
    for(std::size_t i=0;i<scan.count;++i){
        const double stamp=scan.first_ray_stamp_s+static_cast<double>(i)*scan.time_increment_s;
        if(stamp>reference_stamp_s+1e-9)throw std::invalid_argument("deskew reference precedes a ray");
        poses.push_back(integrate({},measured,stamp-reference_stamp_s));
    }
    return poses;
}
} // namespace operator_lab::navigation
