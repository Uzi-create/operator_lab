#pragma once
#include "rgbd_tracker.hpp"
#include "follow_navigation.hpp"

namespace operator_lab::navigation {
// Holds the ORIGINAL camera measurement time. Calling target() never refreshes
// freshness. Pose input must map the fixed tracking frame into the CURRENT base.
class TrackingTargetCache {
public:
    void reset(){point_.reset();stamp_.reset();confirmed_=false;}
    void observe(const rgbd::TrackerResult& result,double measurement_stamp_s){
        if(!std::isfinite(measurement_stamp_s))throw std::invalid_argument("invalid target capture time");
        if(stamp_&&measurement_stamp_s<=*stamp_)throw std::invalid_argument("target frame timestamps must increase");
        stamp_=measurement_stamp_s;confirmed_=false;point_.reset();
        if(result.fresh_observation&&result.confirmed&&result.observed_position_m){
            if(!rgbd::finite(*result.observed_position_m))throw std::invalid_argument("nonfinite tracked point");
            point_=result.observed_position_m;confirmed_=true;
        }
    }
    FollowTarget target(const rgbd::RigidTransform& current_base_from_fixed)const{
        rgbd::validate(current_base_from_fixed);
        if(!point_)return {{},stamp_.value_or(0),false};
        const auto point=rgbd::transform_point(*point_,current_base_from_fixed);
        return {{point.x,point.y},*stamp_,confirmed_};
    }
private:
    std::optional<rgbd::Point3> point_;std::optional<double> stamp_;bool confirmed_=false;
};
struct HeightObstacleConfig {double min_height_m=.05,max_height_m=1.5,max_radius_m=3.,floor_height_m=0;};
// Supplemental camera points must ALREADY be in the current +Z-up base frame.
// This height band assumes a known floor reference; it is not ground estimation.
inline std::vector<Point2> height_obstacles(const std::vector<rgbd::Point3>& points,const HeightObstacleConfig& config={}){
    if(!std::isfinite(config.min_height_m)||!std::isfinite(config.max_height_m)||!std::isfinite(config.max_radius_m)||!std::isfinite(config.floor_height_m)||
       config.min_height_m<0||config.max_height_m<=config.min_height_m||config.max_radius_m<=0)
        throw std::invalid_argument("invalid obstacle height band");
    std::vector<Point2> out;
    for(const auto& point:points){
        if(!rgbd::finite(point))continue;
        const double height=point.z-config.floor_height_m;
        if(height>=config.min_height_m&&height<=config.max_height_m&&std::hypot(point.x,point.y)<=config.max_radius_m)
            out.push_back({point.x,point.y});
    }
    return out;
}
// Adds positive obstacle evidence only; never uses absent camera points to clear space.
inline LaserObservation add_obstacles(LaserObservation laser,const std::vector<Point2>& points,
    double point_capture_stamp_s,double point_reference_stamp_s,double max_age_s=.2,double max_skew_s=.01){
    if(!std::isfinite(point_capture_stamp_s)||!std::isfinite(point_reference_stamp_s)||!std::isfinite(max_skew_s)||
       !std::isfinite(max_age_s)||max_age_s<=0||max_skew_s<0)
        throw std::invalid_argument("invalid supplemental obstacle timing");
    if(!laser.usable||!laser.grid)return laser;
    if(point_capture_stamp_s>laser.reference_stamp_s||laser.reference_stamp_s-point_capture_stamp_s>max_age_s){
        laser.usable=false;laser.grid.reset();laser.reason="supplemental_obstacles_stale_or_future";return laser;
    }
    if(std::abs(point_reference_stamp_s-laser.reference_stamp_s)>max_skew_s){
        laser.usable=false;laser.grid.reset();laser.reason="supplemental_obstacles_wrong_frame_time";return laser;
    }
    for(auto point:points){
        if(!finite(point))throw std::invalid_argument("nonfinite supplemental obstacle");
        laser.grid->mark_occupied(point);laser.obstacle_points.push_back(point);
    }
    return laser;
}
} // namespace operator_lab::navigation
