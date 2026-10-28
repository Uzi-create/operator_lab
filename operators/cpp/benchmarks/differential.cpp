#include <operator_lab/rgbd_tracker.hpp>
#include <operator_lab/follow_navigation.hpp>
#include <operator_lab/depth_target.hpp>
#include <iostream>
#include <cstring>
#include <random>
namespace r=operator_lab::rgbd;namespace n=operator_lab::navigation;
std::uint64_t hash=17;
void add(std::uint64_t x){hash^=x+0x9e3779b97f4a7c15ULL+(hash<<6)+(hash>>2);}
void real(double x){std::uint64_t bits;std::memcpy(&bits,&x,8);add(bits);}
int main(){
    std::mt19937 random(735);n::RobotFootprint robot{.29,.21,.19,.04};n::MotionLimits limits;
    for(int scene=0;scene<30;++scene){
        n::LocalGrid grid({-2.7,-2.6,2.9,2.8,.05});std::vector<n::Point2> points;
        for(std::size_t i=0;i<grid.size();++i){const auto p=grid.center(i);const auto value=random()%20;
            if(value==0){grid.mark_occupied(p);points.push_back(p);}else if(value>2)grid.mark_free(p);}
        n::mark_robot_body(grid,robot);for(std::size_t i=0;i<grid.size();++i)add(static_cast<std::uint64_t>(grid.at(i)));
        for(int i=0;i<400;++i){
            const n::Pose2 pose{(static_cast<int>(random()%700)-350)*.01,(static_cast<int>(random()%700)-350)*.01,(random()%629)*.01};
            const auto result=n::check_footprint(grid,pose,robot,(random()%30)*.01);add(static_cast<std::uint64_t>(result.state));
            add(result.blocking_cell.has_value());if(result.blocking_cell){real(result.blocking_cell->x);real(result.blocking_cell->y);}
            add(n::footprint_intersects_cell({.3,-.2},.025,pose,robot,.1));
        }
        for(int i=0;i<15;++i){n::TrajectoryConfig config;config.obstacle_speed_bound_mps=i%2?.5:0;
            const auto result=n::check_trajectory(grid,robot,limits,{.1,.01,.2},{.12,.02,.25},config,.01);
            add(result.clear);add(static_cast<std::uint64_t>(result.collision));real(result.time_s);real(result.end_pose.x);real(result.end_pose.y);real(result.end_pose.yaw);add(result.path.size());
        }
    }
    std::cout<<"geometry "<<hash<<"\n";
    for(int trial=0;trial<40;++trial){
        std::vector<float> ranges(721);for(auto& value:ranges){const auto v=random()%40;value=v==0?std::numeric_limits<float>::infinity():(v==1?std::numeric_limits<float>::quiet_NaN():static_cast<float>(v)*.2f);}
        n::LaserScanView scan{ranges.data(),ranges.size(),trial%2?-n::pi:n::pi,(trial%2?1:-1)*2*n::pi/720,.15,10,1,0};
        n::LaserConfig config;config.positive_infinity_is_clear=trial%2;const n::Pose2 pose{.12,-.03,.7};
        auto result=n::laser_to_grid(scan,pose,1,config);add(result.usable);add(result.usable_rays);add(result.obstacle_returns);
        if(result.grid)for(std::size_t i=0;i<result.grid->size();++i)add(static_cast<std::uint64_t>(result.grid->at(i)));
        for(auto point:result.obstacle_points){real(point.x);real(point.y);}
    }
    std::cout<<"laser "<<hash<<"\n";
    for(int trial=0;trial<45;++trial){
        const std::size_t w=trial%2?80:21,h=trial%2?64:17;std::vector<std::uint16_t> depth(w*h);
        for(auto& value:depth)value=random()%30?static_cast<std::uint16_t>(2000+random()%100):0;
        r::DepthView<std::uint16_t> view(depth.data(),depth.size(),w,h,.001);r::Intrinsics k{w,h,80,80,w/2.,h/2.};
        r::DepthRegionConfig config;config.min_pixels=1;config.connectivity=trial%3?4:8;
        auto regions=r::extract_depth_regions(view,k,{0,0,w,h},config);add(regions.size());
        for(const auto& region:regions){add(region.label);add(region.bounds.x);add(region.bounds.y);add(region.bounds.width);add(region.bounds.height);real(region.depth_m);real(region.mad_m);real(region.point_m.x);real(region.point_m.y);real(region.point_m.z);for(auto p:region.pixels){add(p.u);add(p.v);}}
        auto band=r::estimate_depth_target(view,k,{0,0,w,h});add(static_cast<std::uint64_t>(band.state));real(band.depth_m.value_or(0));real(band.mad_m.value_or(0));
    }
    std::cout<<"depth "<<hash<<"\n";
    for(int trial=0;trial<80;++trial){
        std::vector<r::TrackCandidate> candidates;for(std::size_t i=0;i<30;++i){std::vector<double> descriptor(128);
            for(auto& value:descriptor)value=(static_cast<int>(random()%200)-100)*.01;
            if(i==3&&trial%2)descriptor[7]=std::numeric_limits<double>::quiet_NaN();
            candidates.push_back({i,{(static_cast<int>(random()%500)-250)*.01,0,2},std::move(descriptor)});}
        r::AssociationQuery query{{0,0,2},{.1,.1,.1},candidates.front().appearance};r::AssociationConfig config;config.max_appearance_distance=2;
        const auto result=r::associate_target(candidates,query,config);add(static_cast<std::uint64_t>(result.state));add(result.candidate_index.value_or(999));real(result.best_score);real(result.second_score);
    }
    std::cout<<"association "<<hash<<"\n";
    n::LocalGrid grid;grid.fill(n::Cell::free);std::vector<n::Point2> points;
    for(int i=0;i<1000;++i){const double angle=2*n::pi*i/1000;n::Point2 p{2.5*std::cos(angle),2.5*std::sin(angle)};grid.mark_occupied(p);points.push_back(p);}
    for(int trial=0;trial<30;++trial){
        n::AvoidanceConfig config;config.trajectory.obstacle_speed_bound_mps=trial%2?.5:0;
        const auto plan=n::plan_local_motion(grid,points,robot,limits,{.2,0,.1},{.3,.01,.1},{1.2,.1},config);
        add(plan.found);add(plan.evaluated);add(plan.rejected);real(plan.score);real(plan.command.x);real(plan.command.y);real(plan.command.yaw);add(plan.trajectory.path.size());
        for(auto p:plan.trajectory.path){real(p.x);real(p.y);real(p.yaw);}
    }
    std::cout<<"reference_digest="<<hash<<'\n';
}
