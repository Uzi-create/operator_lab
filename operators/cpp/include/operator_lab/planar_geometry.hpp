#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace operator_lab::navigation {
constexpr double pi = 3.14159265358979323846;
struct Point2 { double x=0,y=0; };
struct Pose2 { double x=0,y=0,yaw=0; };
struct Velocity2 { double x=0,y=0,yaw=0; };
inline bool finite(Point2 p){return std::isfinite(p.x)&&std::isfinite(p.y);}
inline bool finite(Pose2 p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.yaw);}
inline bool finite(Velocity2 p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.yaw);}
inline double speed(Velocity2 v){return std::hypot(v.x,v.y);}
inline double distance(Point2 a,Point2 b){return std::hypot(a.x-b.x,a.y-b.y);}
inline bool stopped(Velocity2 v,double epsilon=1e-8){return speed(v)<=epsilon&&std::abs(v.yaw)<=epsilon;}
inline Point2 transform_point(Point2 point,Pose2 destination_from_source){
    if(!finite(point)||!finite(destination_from_source))throw std::invalid_argument("nonfinite planar transform");
    const double c=std::cos(destination_from_source.yaw),s=std::sin(destination_from_source.yaw);
    Point2 out{c*point.x-s*point.y+destination_from_source.x,s*point.x+c*point.y+destination_from_source.y};
    if(!finite(out))throw std::overflow_error("planar transform overflow");return out;
}
inline Pose2 inverse(Pose2 pose){
    if(!finite(pose))throw std::invalid_argument("nonfinite planar pose");
    const double c=std::cos(pose.yaw),s=std::sin(pose.yaw);
    Pose2 out{-c*pose.x-s*pose.y,s*pose.x-c*pose.y,-pose.yaw};
    if(!finite(out))throw std::overflow_error("planar inverse overflow");return out;
}
// Exact SE(2) integration for constant BODY-frame velocity, including lateral motion.
inline Pose2 integrate(Pose2 initial,Velocity2 body,double dt){
    if(!finite(initial)||!finite(body)||!std::isfinite(dt))throw std::invalid_argument("invalid planar integration");
    const double theta=body.yaw*dt;
    if(!std::isfinite(theta))throw std::overflow_error("integration angle overflow");
    const double sinc=std::abs(theta)<1e-6?1-theta*theta/6:std::sin(theta)/theta;
    const double cosc=std::abs(theta)<1e-6?theta/2-theta*theta*theta/24:2*std::pow(std::sin(theta/2),2)/theta;
    const Point2 local{dt*(sinc*body.x-cosc*body.y),dt*(cosc*body.x+sinc*body.y)};
    const auto point=transform_point(local,initial);
    return {point.x,point.y,std::remainder(initial.yaw+theta,2*pi)};
}
struct RobotFootprint {
    // REQUIRED geometry in meters relative to the robot's chosen base origin.
    // Zero defaults are intentionally invalid; do not guess the R550 footprint.
    double front=0,back=0,half_width=0,margin=0;
};
inline void validate(RobotFootprint f){
    for(double value:{f.front,f.back,f.half_width,f.margin})
        if(!std::isfinite(value)||value<0||value>5)throw std::invalid_argument("invalid footprint");
    if(f.front<=0||f.back<=0||f.half_width<=0)throw std::invalid_argument("footprint must be explicitly supplied");
}
inline double corner_radius(RobotFootprint f){return std::hypot(std::max(f.front,f.back)+f.margin,f.half_width+f.margin);}
struct MotionLimits {
    // Example algorithm limits, NOT measured R550 specifications.
    double max_linear=.4,max_lateral=.3,max_yaw=1.;
    double accel_linear=.5,accel_yaw=1.,brake_linear=.5,brake_yaw=1.;
};
inline void validate(MotionLimits l){
    for(double value:{l.max_linear,l.max_lateral,l.max_yaw,l.accel_linear,l.accel_yaw,l.brake_linear,l.brake_yaw})
        if(!std::isfinite(value)||value<0||value>10)throw std::invalid_argument("invalid motion limit");
    if(l.max_linear<=0||l.max_yaw<=0||l.accel_linear<.001||l.accel_yaw<.001||
       l.brake_linear<.001||l.brake_yaw<.001)throw std::invalid_argument("motion and braking limits must be positive");
}
inline Velocity2 bound_velocity(Velocity2 v,MotionLimits limits){
    if(!finite(v))throw std::invalid_argument("nonfinite velocity");validate(limits);
    v.y=std::clamp(v.y,-limits.max_lateral,limits.max_lateral);
    const double norm=speed(v);
    if(norm>limits.max_linear){v.x*=limits.max_linear/norm;v.y*=limits.max_linear/norm;}
    v.yaw=std::clamp(v.yaw,-limits.max_yaw,limits.max_yaw);return v;
}
inline Velocity2 approach_velocity(Velocity2 current,Velocity2 desired,MotionLimits limits,double dt){
    if(!finite(current)||!std::isfinite(dt)||dt<=0||dt>1)throw std::invalid_argument("invalid velocity smoothing input");
    desired=bound_velocity(desired,limits);
    double dx=desired.x-current.x,dy=desired.y-current.y;
    const double change=std::hypot(dx,dy),allowed=limits.accel_linear*dt;
    if(change>allowed){dx*=allowed/change;dy*=allowed/change;}
    return {current.x+dx,current.y+dy,current.yaw+std::clamp(desired.yaw-current.yaw,-limits.accel_yaw*dt,limits.accel_yaw*dt)};
}
inline Velocity2 brake_velocity(Velocity2 current,MotionLimits limits,double dt){
    validate(limits);
    if(!finite(current)||!std::isfinite(dt)||dt<0)throw std::invalid_argument("invalid braking input");
    const double norm=speed(current),remaining=std::max(0.,norm-limits.brake_linear*dt);
    const double scale=norm>0?remaining/norm:0;
    return {current.x*scale,current.y*scale,std::copysign(std::max(0.,std::abs(current.yaw)-limits.brake_yaw*dt),current.yaw)};
}
} // namespace operator_lab::navigation
