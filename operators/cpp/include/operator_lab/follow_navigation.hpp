#pragma once
#include "follow_control.hpp"
#include "local_avoidance.hpp"

namespace operator_lab::navigation {
struct VelocityObservation {Velocity2 velocity;double stamp_s=0;};
struct NavigationConfig {
    FollowConfig follow;AvoidanceConfig avoidance;
    double max_scan_age_s=.2,max_velocity_age_s=.15,max_grid_reference_skew_s=.01;
    std::size_t clear_scan_count=3;double clear_duration_s=.15;
};
struct NavigationResult {
    Velocity2 command,desired;
    bool moving_allowed=false,braking_path_clear=false;
    std::string reason;
    LocalPlan plan;
};
// Final velocity arbiter. Emergency/invalid inputs request zero directly, never
// through a smoother that could delay braking. Zero command is NOT an instant stop.
class FollowNavigator {
public:
    FollowNavigator(RobotFootprint robot,MotionLimits limits,NavigationConfig config={})
        :robot_(robot),limits_(limits),config_(config){
        validate(robot_);validate(limits_);validate(config_.follow);validate(config_.avoidance);
        for(double value:{config_.max_scan_age_s,config_.max_velocity_age_s,config_.max_grid_reference_skew_s,config_.clear_duration_s})
            if(!std::isfinite(value)||value<0)throw std::invalid_argument("invalid navigation timing");
        if(config_.max_scan_age_s<=0||config_.max_velocity_age_s<=0||!config_.clear_scan_count)
            throw std::invalid_argument("invalid navigation confirmation");
    }
    void reset(){last_now_.reset();last_scan_seen_.reset();last_clear_scan_.reset();clear_started_.reset();clear_count_=0;side_=0;}
    NavigationResult step(FollowTarget target,const LaserObservation& laser,VelocityObservation measured,
                          double now_s,bool emergency_stop=false){
        if(!std::isfinite(now_s))return halt("nonfinite_control_time");
        if(last_now_&&now_s<*last_now_){reset();return halt("control_clock_rewind");}
        const double control_dt=last_now_?std::min(config_.avoidance.trajectory.control_dt_s,now_s-*last_now_):config_.avoidance.trajectory.control_dt_s;
        last_now_=now_s;
        if(control_dt<.001)return halt("control_interval_too_short");
        if(emergency_stop)return halt("emergency_stop");
        if(!finite(measured.velocity)||!std::isfinite(measured.stamp_s)||measured.stamp_s>now_s||
           now_s-measured.stamp_s>config_.max_velocity_age_s)return halt("invalid_or_stale_velocity");
        if(!laser.usable||!laser.grid)return halt("laser_unusable");
        if(!std::isfinite(laser.first_stamp_s)||!std::isfinite(laser.last_stamp_s)||!std::isfinite(laser.reference_stamp_s)||
           laser.first_stamp_s>laser.last_stamp_s||laser.last_stamp_s>now_s+1e-9||now_s-laser.first_stamp_s>config_.max_scan_age_s)
            return halt("laser_stale_or_future");
        if(laser.reference_stamp_s>now_s+1e-9||std::abs(now_s-laser.reference_stamp_s)>config_.max_grid_reference_skew_s)return halt("grid_reference_outdated");
        if(last_scan_seen_&&laser.first_stamp_s<*last_scan_seen_)return halt("out_of_order_scan");
        last_scan_seen_=laser.first_stamp_s;
        if(measured.velocity.x<0||speed(measured.velocity)>limits_.max_linear+1e-6||std::abs(measured.velocity.y)>limits_.max_lateral+1e-6||
           std::abs(measured.velocity.yaw)>limits_.max_yaw+1e-6)return halt("measured_overspeed");
        if(!finite(target.point_base)||!std::isfinite(target.stamp_s))return halt("invalid_target");
        for(auto point:laser.obstacle_points)if(!finite(point))return halt("invalid_obstacle_points");
        const auto follow=follow_command(target,now_s,limits_,config_.follow);
        if(!follow.target_usable)return halt(follow.reason);
        auto grid=*laser.grid;mark_robot_body(grid,robot_);
        auto local_config=config_.avoidance;local_config.trajectory.control_dt_s=control_dt;
        const double age=std::max(0.,now_s-laser.first_stamp_s);
        NavigationResult out;out.desired=follow.velocity;
        const auto braking=check_trajectory(grid,robot_,limits_,measured.velocity,{},local_config.trajectory,age,true);
        out.braking_path_clear=braking.clear;
        if(!braking.clear){auto stop=halt(braking.collision==CollisionState::unknown?"braking_space_unknown":"braking_path_obstructed");return stop;}
        const double range=std::hypot(target.point_base.x,target.point_base.y);
        const double scale=range>0?std::max(0.,range-config_.follow.desired_distance_m)/range:0;
        const Point2 goal{target.point_base.x*scale,target.point_base.y*scale};
        out.plan=plan_local_motion(grid,laser.obstacle_points,robot_,limits_,measured.velocity,follow.velocity,goal,
                                   local_config,age,side_);
        if(!out.plan.found){auto stop=halt(out.plan.reason);stop.plan=std::move(out.plan);stop.braking_path_clear=true;return stop;}
        side_=out.plan.goal.side;
        if(!last_clear_scan_||laser.first_stamp_s>*last_clear_scan_){
            if(!clear_started_)clear_started_=now_s;
            last_clear_scan_=laser.first_stamp_s;
            if(clear_count_<config_.clear_scan_count)++clear_count_;
        }else if(laser.first_stamp_s<*last_clear_scan_)return halt("out_of_order_scan");
        if(clear_count_<config_.clear_scan_count||now_s-*clear_started_<config_.clear_duration_s){
            out.reason="waiting_clear_scans";return out;
        }
        // Recheck the selected candidate as the final command stage.
        const auto final_check=check_trajectory(grid,robot_,limits_,measured.velocity,out.plan.command,local_config.trajectory,age);
        if(!final_check.clear)return halt("final_collision_check_failed");
        out.command=out.plan.command;out.moving_allowed=true;out.reason=out.plan.reason;return out;
    }
private:
    RobotFootprint robot_;MotionLimits limits_;NavigationConfig config_;
    std::optional<double> last_now_,last_scan_seen_,last_clear_scan_,clear_started_;std::size_t clear_count_=0;int side_=0;
    NavigationResult halt(std::string reason){
        last_clear_scan_.reset();clear_started_.reset();clear_count_=0;side_=0;
        NavigationResult out;out.reason=std::move(reason);return out;
    }
};
} // namespace operator_lab::navigation
