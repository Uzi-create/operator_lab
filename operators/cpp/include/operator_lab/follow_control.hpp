#pragma once
#include "planar_geometry.hpp"
#include <string>

namespace operator_lab::navigation {
struct FollowTarget {Point2 point_base;double stamp_s=0;bool confirmed=false;};
struct FollowConfig {
    double desired_distance_m=1.2,distance_deadband_m=.1,bearing_deadband_rad=.05;
    double distance_gain=.6,bearing_gain=1.2,turn_first_angle_rad=.65,max_target_age_s=.2,max_target_range_m=8.;
};
struct FollowCommand {Velocity2 velocity;bool target_usable=false;std::string reason;};
inline void validate(const FollowConfig& c){
    for(double value:{c.desired_distance_m,c.distance_deadband_m,c.bearing_deadband_rad,c.distance_gain,
                      c.bearing_gain,c.turn_first_angle_rad,c.max_target_age_s,c.max_target_range_m})
        if(!std::isfinite(value)||value<0)throw std::invalid_argument("invalid follow configuration");
    if(c.desired_distance_m<=0||c.distance_deadband_m>=c.desired_distance_m||c.turn_first_angle_rad<=0||
       c.turn_first_angle_rad>pi||c.max_target_age_s<=0||c.max_target_range_m<=c.desired_distance_m||
       c.max_target_range_m>1000||c.distance_gain>100||c.bearing_gain>100)throw std::invalid_argument("invalid follow bounds");
}
// P feedback for distance/bearing, with deadbands, turn-first behavior and no
// automatic reverse. Lateral movement is left to the local avoidance planner.
inline FollowCommand follow_command(FollowTarget target,double now_s,MotionLimits limits,const FollowConfig& config={}){
    validate(config);validate(limits);
    if(!finite(target.point_base)||!std::isfinite(target.stamp_s)||!std::isfinite(now_s))
        throw std::invalid_argument("nonfinite follow target");
    if(!target.confirmed)return {{},false,"target_unconfirmed"};
    if(target.stamp_s>now_s)return {{},false,"future_target"};
    if(now_s-target.stamp_s>config.max_target_age_s)return {{},false,"stale_target"};
    const double range=std::hypot(target.point_base.x,target.point_base.y);
    if(!std::isfinite(range)||range>config.max_target_range_m)return {{},false,"target_range_untrusted"};
    if(range<=config.desired_distance_m+config.distance_deadband_m)return {{},true,"distance_hold"};
    const double angle=std::atan2(target.point_base.y,target.point_base.x);
    Velocity2 desired;
    if(std::abs(angle)>config.bearing_deadband_rad)desired.yaw=config.bearing_gain*angle;
    if(std::abs(angle)<config.turn_first_angle_rad)
        desired.x=config.distance_gain*(range-config.desired_distance_m)*std::max(0.,std::cos(angle));
    return {bound_velocity(desired,limits),true,std::abs(angle)>=config.turn_first_angle_rad?"turn_first":"follow_target"};
}
} // namespace operator_lab::navigation
