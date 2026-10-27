#pragma once
#include "laser_grid.hpp"

namespace operator_lab::navigation {
enum class CollisionState {clear,occupied,unknown};
struct FootprintCheck {CollisionState state=CollisionState::clear;std::optional<Point2> blocking_cell;};
// SAT intersection between an oriented, padded robot rectangle and a grid cell.
inline bool footprint_intersects_cell(Point2 cell,double half_cell,Pose2 pose,RobotFootprint robot,double extra=0){
    const double c=std::cos(pose.yaw),s=std::sin(pose.yaw);
    const double center_offset=(robot.front-robot.back)/2;
    const Point2 center{pose.x+c*center_offset,pose.y+s*center_offset};
    const double a=(robot.front+robot.back)/2+robot.margin+extra,b=robot.half_width+robot.margin+extra;
    const double dx=cell.x-center.x,dy=cell.y-center.y;
    if(std::abs(dx)>a*std::abs(c)+b*std::abs(s)+half_cell ||
       std::abs(dy)>a*std::abs(s)+b*std::abs(c)+half_cell)return false;
    const double cell_projection=half_cell*(std::abs(c)+std::abs(s));
    return std::abs(c*dx+s*dy)<=a+cell_projection&&std::abs(-s*dx+c*dy)<=b+cell_projection;
}
inline FootprintCheck check_footprint(const LocalGrid& grid,Pose2 pose,RobotFootprint robot,double extra=0){
    validate(robot);
    if(!finite(pose)||!std::isfinite(extra)||extra<0||extra>100)throw std::invalid_argument("invalid footprint check pose or padding");
    const double c=std::cos(pose.yaw),s=std::sin(pose.yaw),offset=(robot.front-robot.back)/2;
    const double a=(robot.front+robot.back)/2+robot.margin+extra,b=robot.half_width+robot.margin+extra;
    const Point2 center{pose.x+c*offset,pose.y+s*offset};
    const double ex=a*std::abs(c)+b*std::abs(s),ey=a*std::abs(s)+b*std::abs(c);
    const auto& g=grid.config();
    if(center.x-ex<g.xmin||center.y-ey<g.ymin||center.x+ex>=g.xmax||center.y+ey>=g.ymax)
        return {CollisionState::unknown,std::nullopt};
    const auto xmin=static_cast<std::size_t>(std::floor((center.x-ex-g.xmin)/g.resolution));
    const auto xmax=std::min(grid.width()-1,static_cast<std::size_t>(std::floor((center.x+ex-g.xmin)/g.resolution)));
    const auto ymin=static_cast<std::size_t>(std::floor((center.y-ey-g.ymin)/g.resolution));
    const auto ymax=std::min(grid.height()-1,static_cast<std::size_t>(std::floor((center.y+ey-g.ymin)/g.resolution)));
    FootprintCheck out;
    for(std::size_t y=ymin;y<=ymax;++y)for(std::size_t x=xmin;x<=xmax;++x){
        const auto index=y*grid.width()+x;const auto point=grid.center(index);
        if(grid.at(index)==Cell::free||!footprint_intersects_cell(point,g.resolution/2,pose,robot,extra))continue;
        if(grid.at(index)==Cell::occupied)return {CollisionState::occupied,point};
        out={CollisionState::unknown,point};
    }
    return out;
}
// The robot already occupies its own body, but never erase an obstacle return.
// Only cells ENTIRELY inside the unpadded current body are marked known-free.
inline void mark_robot_body(LocalGrid& grid,RobotFootprint robot){
    validate(robot);const double h=grid.config().resolution/2;
    for(std::size_t i=0;i<grid.size();++i){
        const auto p=grid.center(i);
        if(p.x-h>=-robot.back&&p.x+h<=robot.front&&p.y-h>=-robot.half_width&&p.y+h<=robot.half_width)
            grid.mark_free(p);
    }
}
struct TrajectoryConfig {
    double control_dt_s=.1,horizon_s=1.2,reaction_s=.1,time_step_s=.05;
    // Example bounded-motion model, NOT a measured pedestrian speed guarantee.
    // Use zero only for explicitly static synthetic/controlled scenes.
    double tracking_margin_m=.02,obstacle_speed_bound_mps=.5;
};
inline void validate(const TrajectoryConfig& c){
    for(double value:{c.control_dt_s,c.horizon_s,c.reaction_s,c.time_step_s,c.tracking_margin_m,c.obstacle_speed_bound_mps})
        if(!std::isfinite(value)||value<0)throw std::invalid_argument("invalid trajectory configuration");
    if(c.control_dt_s<.001||c.control_dt_s>1||c.horizon_s<c.control_dt_s||c.horizon_s>5||
       c.reaction_s>1||c.time_step_s<.005||c.time_step_s>.2||c.tracking_margin_m>1||c.obstacle_speed_bound_mps>3)
        throw std::invalid_argument("unsupported trajectory limits");
}
struct TrajectoryResult {
    bool clear=false;CollisionState collision=CollisionState::clear;
    double time_s=0;Pose2 end_pose;std::optional<Point2> blocking_cell;
    std::vector<Pose2> path;
};
// Sampled body-twist rollout includes latency, acceleration-limited approach,
// commanded motion, and a complete discretized braking tail. Swept padding covers
// motion between samples. This is a model check, not a physical safety certificate.
inline TrajectoryResult check_trajectory(const LocalGrid& grid,RobotFootprint robot,MotionLimits limits,
    Velocity2 current,Velocity2 command,const TrajectoryConfig& config={},double scan_age_s=0,bool braking_only=false){
    validate(robot);validate(limits);validate(config);
    if(!finite(current)||!finite(command)||!std::isfinite(scan_age_s)||scan_age_s<0||scan_age_s>10||
       speed(current)>10||std::abs(current.yaw)>10)throw std::invalid_argument("invalid trajectory motion input");
    command=bound_velocity(command,limits);
    if(!braking_only&&(std::hypot(command.x-current.x,command.y-current.y)>limits.accel_linear*config.control_dt_s+1e-8||
       std::abs(command.yaw-current.yaw)>limits.accel_yaw*config.control_dt_s+1e-8))
        throw std::invalid_argument("trajectory command is outside acceleration window");
    TrajectoryResult out;Pose2 pose;Velocity2 velocity=current;out.path.push_back(pose);
    auto inspect=[&](double sweep){
        const double padding=config.tracking_margin_m+sweep+config.obstacle_speed_bound_mps*(scan_age_s+out.time_s);
        if(padding>100){out.collision=CollisionState::unknown;return false;}
        const auto state=check_footprint(grid,pose,robot,padding);
        out.collision=state.state;out.blocking_cell=state.blocking_cell;
        return state.state==CollisionState::clear;
    };
    if(!inspect(0))return out;
    std::size_t iterations=0;
    auto advance_phase=[&](double duration,bool brake,bool approach){
        double elapsed=0;
        while(elapsed<duration-1e-12){
            if(++iterations>10000){out.collision=CollisionState::unknown;return false;}
            const double point_speed=speed(velocity)+corner_radius(robot)*std::abs(velocity.yaw);
            double dt=std::min(config.time_step_s,duration-elapsed);
            if(point_speed>0)dt=std::min(dt,grid.config().resolution/(2*point_speed));
            if(!inspect(point_speed*dt))return false;
            pose=integrate(pose,velocity,dt);elapsed+=dt;out.time_s+=dt;out.path.push_back(pose);out.end_pose=pose;
            if(brake)velocity=brake_velocity(velocity,limits,dt);
            else if(approach)velocity=approach_velocity(velocity,command,limits,dt);
        }
        return inspect(0);
    };
    if(!advance_phase(config.reaction_s,false,false))return out;
    if(!braking_only){
        if(!advance_phase(config.control_dt_s,false,true))return out;
        // Finish a reachable command exactly after the control interval; the planner
        // limits candidates to the dynamic window before calling this function.
        velocity=command;
        if(!advance_phase(config.horizon_s-config.control_dt_s,false,false))return out;
    }
    const double stop_time=std::max(speed(velocity)/limits.brake_linear,std::abs(velocity.yaw)/limits.brake_yaw);
    if(stop_time>30){out.collision=CollisionState::unknown;return out;}
    if(!advance_phase(stop_time+config.time_step_s,true,false))return out;
    out.clear=true;return out;
}
inline double point_clearance(Point2 point,Pose2 pose,RobotFootprint robot){
    const auto p=transform_point(point,inverse(pose));
    const double dx=std::max({-robot.back-robot.margin-p.x,0.,p.x-robot.front-robot.margin});
    const double dy=std::max(0.,std::abs(p.y)-robot.half_width-robot.margin);
    return std::hypot(dx,dy);
}
} // namespace operator_lab::navigation
