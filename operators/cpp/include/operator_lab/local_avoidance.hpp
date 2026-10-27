#pragma once
#include "collision_check.hpp"

namespace operator_lab::navigation {
struct AvoidanceConfig {
    TrajectoryConfig trajectory;
    std::size_t linear_samples=5,lateral_samples=3,yaw_samples=5;
    bool allow_lateral=true,allow_detour=false;
    double detour_clearance_m=.15,lookahead_m=2.,clearance_weight=.08;
};
inline void validate(const AvoidanceConfig& c){
    validate(c.trajectory);
    if(c.linear_samples<2||c.linear_samples>9||c.lateral_samples<2||c.lateral_samples>9||c.yaw_samples<2||c.yaw_samples>9||
       !std::isfinite(c.detour_clearance_m)||c.detour_clearance_m<0||c.detour_clearance_m>1||
       !std::isfinite(c.lookahead_m)||c.lookahead_m<=0||c.lookahead_m>5||
       !std::isfinite(c.clearance_weight)||c.clearance_weight<0||c.clearance_weight>10)
        throw std::invalid_argument("invalid local avoidance configuration");
}
struct LocalGoal {Point2 point;bool obstructed=false,detour=false;int side=0;std::string reason;};
inline LocalGoal select_local_goal(const LocalGrid& grid,const std::vector<Point2>& obstacle_points,
    Point2 goal,RobotFootprint robot,const AvoidanceConfig& config,int preferred_side=0){
    validate(robot);validate(config);
    if(!finite(goal))throw std::invalid_argument("invalid local goal");
    const double range=std::hypot(goal.x,goal.y);
    LocalGoal out{goal,false,false,0,"direct_goal"};
    if(range<1e-8)return out;
    const Point2 direction{goal.x/range,goal.y/range},normal{-direction.y,direction.x};
    std::optional<Point2> first;
    for(double d=0;d<=std::min(range,config.lookahead_m);d+=grid.config().resolution/2){
        const auto check=check_footprint(grid,{direction.x*d,direction.y*d,0},robot);
        if(check.state==CollisionState::unknown){out.obstructed=true;out.reason="direct_space_unknown";return out;}
        if(check.state==CollisionState::occupied){first=check.blocking_cell;break;}
    }
    if(!first)return out;
    out.obstructed=true;out.reason="direct_path_blocked";
    if(!config.allow_detour)return out;
    const double along=first->x*direction.x+first->y*direction.y;
    double low=first->x*normal.x+first->y*normal.y,high=low;
    for(auto point:obstacle_points){
        if(!finite(point))throw std::invalid_argument("invalid obstacle point");
        if(!grid.index(point))continue; // Far returns outside this local map do not define a bypass wall.
        const double d=point.x*direction.x+point.y*direction.y;
        if(std::abs(d-along)>corner_radius(robot)+config.detour_clearance_m)continue;
        const double cross=point.x*normal.x+point.y*normal.y;low=std::min(low,cross);high=std::max(high,cross);
    }
    const double padding=corner_radius(robot)+config.detour_clearance_m+grid.config().resolution;
    double best=std::numeric_limits<double>::infinity();
    for(int side:{1,-1}){
        const double lateral=side>0?high+padding:low-padding;
        // Approach a near-side corner first; once alongside, aim past the obstacle.
        const bool alongside=(side>0?high< -padding*.7:low>padding*.7);
        const double longitudinal=alongside?along+robot.back+config.detour_clearance_m:
                                             std::max(0.,along-robot.front-config.detour_clearance_m);
        const Point2 candidate{direction.x*longitudinal+normal.x*lateral,
                               direction.y*longitudinal+normal.y*lateral};
        if(check_footprint(grid,{candidate.x,candidate.y,0},robot).state!=CollisionState::clear)continue;
        const double score=std::hypot(candidate.x,candidate.y)+.2*distance(candidate,goal)+
                           (preferred_side&&side!=preferred_side?.4:0);
        if(score<best){best=score;out.point=candidate;out.detour=true;out.side=side;out.reason="local_detour";}
    }
    return out;
}
struct LocalPlan {
    bool found=false;Velocity2 command;TrajectoryResult trajectory;LocalGoal goal;
    std::size_t evaluated=0,rejected=0;double score=std::numeric_limits<double>::infinity();
    std::string reason;
};
namespace avoidance_detail {
inline std::vector<double> samples(double low,double high,std::size_t count,double desired){
    if(low>high)throw std::invalid_argument("empty velocity sampling window");
    std::vector<double> out;
    for(std::size_t i=0;i<count;++i)out.push_back(low+(high-low)*static_cast<double>(i)/static_cast<double>(count-1));
    out.push_back(std::clamp(desired,low,high));if(low<=0&&high>=0)out.push_back(0);
    std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;
}
}
// Local dynamic-window sampling with a geometric detour waypoint. It can stop
// in dead ends; no global map, global route completeness or pedestrian intent model.
inline LocalPlan plan_local_motion(const LocalGrid& grid,const std::vector<Point2>& obstacle_points,
    RobotFootprint robot,MotionLimits limits,Velocity2 current,Velocity2 desired,Point2 goal,
    const AvoidanceConfig& config={},double scan_age_s=0,int preferred_side=0){
    validate(robot);validate(limits);validate(config);
    if(!finite(current)||!finite(desired)||!finite(goal))throw std::invalid_argument("invalid local planner input");
    if(current.x<0||speed(current)>limits.max_linear+1e-8||std::abs(current.y)>limits.max_lateral+1e-8||std::abs(current.yaw)>limits.max_yaw+1e-8){
        LocalPlan out;out.reason="motion_outside_supported_limits";return out;
    }
    LocalPlan out;out.goal=select_local_goal(grid,obstacle_points,goal,robot,config,preferred_side);
    if(out.goal.obstructed&&!out.goal.detour){out.reason=out.goal.reason;return out;}
    const double dv=limits.accel_linear*config.trajectory.control_dt_s,dw=limits.accel_yaw*config.trajectory.control_dt_s;
    const auto xs=avoidance_detail::samples(std::max(0.,current.x-dv),std::min(limits.max_linear,current.x+dv),config.linear_samples,desired.x);
    const auto ys=config.allow_lateral?avoidance_detail::samples(std::max(-limits.max_lateral,current.y-dv),
        std::min(limits.max_lateral,current.y+dv),config.lateral_samples,desired.y):std::vector<double>{0};
    const auto ws=avoidance_detail::samples(std::max(-limits.max_yaw,current.yaw-dw),
        std::min(limits.max_yaw,current.yaw+dw),config.yaw_samples,desired.yaw);
    std::vector<Velocity2> candidates;
    for(double x:xs)for(double y:ys)for(double w:ws)candidates.push_back({x,y,w});
    const double goal_range=std::hypot(out.goal.point.x,out.goal.point.y);
    if(out.goal.detour&&goal_range>0){
        Velocity2 toward{limits.max_linear*out.goal.point.x/goal_range,
                         config.allow_lateral?limits.max_linear*out.goal.point.y/goal_range:0,desired.yaw};
        if(!config.allow_lateral)toward.yaw=std::atan2(out.goal.point.y,out.goal.point.x);
        candidates.push_back(approach_velocity(current,toward,limits,config.trajectory.control_dt_s));
    }
    candidates.push_back(approach_velocity(current,desired,limits,config.trajectory.control_dt_s));
    for(auto command:candidates){
        if(command.x<0||speed(command)>limits.max_linear+1e-8||std::abs(command.y)>limits.max_lateral+1e-8||
           std::hypot(command.x-current.x,command.y-current.y)>dv+1e-8||std::abs(command.yaw-current.yaw)>dw+1e-8)continue;
        ++out.evaluated;
        auto trajectory=check_trajectory(grid,robot,limits,current,command,config.trajectory,scan_age_s);
        if(!trajectory.clear){++out.rejected;continue;}
        double clearance=2.;
        for(auto point:obstacle_points)clearance=std::min(clearance,point_clearance(point,trajectory.end_pose,robot));
        const double velocity_error=std::hypot(command.x-desired.x,command.y-desired.y)/limits.max_linear;
        const double score=4*distance({trajectory.end_pose.x,trajectory.end_pose.y},out.goal.point)+
            (out.goal.detour?.25:1.)*velocity_error+.1*std::abs(command.yaw-desired.yaw)+config.clearance_weight/(clearance+.1);
        if(score<out.score){out.score=score;out.found=true;out.command=command;out.trajectory=std::move(trajectory);}
    }
    out.reason=out.found?(out.goal.detour?"safe_local_detour":"safe_follow_command"):"no_safe_velocity";
    return out;
}
} // namespace operator_lab::navigation
