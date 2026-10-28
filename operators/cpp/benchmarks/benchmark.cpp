#include <operator_lab/rgbd_tracker.hpp>
#include <operator_lab/depth_target.hpp>
#include <operator_lab/follow_navigation.hpp>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <numeric>

// Counts allocations requested through replaceable C++ new, not RSS or peak live
// memory. Fixture setup and statistics storage are outside the measured scope.
namespace allocation_probe {
thread_local bool active=false;
thread_local std::size_t calls=0,bytes=0;
void note(std::size_t n){if(active){++calls;bytes+=n;}}
}
void* operator new(std::size_t n){if(auto p=std::malloc(n?n:1)){allocation_probe::note(n);return p;}throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
void* operator new(std::size_t n,std::align_val_t alignment){
    void* p=nullptr;if(posix_memalign(&p,static_cast<std::size_t>(alignment),n?n:1))throw std::bad_alloc();
    allocation_probe::note(n);return p;
}
void* operator new[](std::size_t n,std::align_val_t a){return ::operator new(n,a);}
void operator delete(void* p,std::align_val_t)noexcept{std::free(p);}
void operator delete[](void* p,std::align_val_t)noexcept{std::free(p);}
void operator delete(void* p,std::size_t,std::align_val_t)noexcept{std::free(p);}
void operator delete[](void* p,std::size_t,std::align_val_t)noexcept{std::free(p);}

namespace r=operator_lab::rgbd;
namespace n=operator_lab::navigation;
volatile std::uint64_t consumed=0;
std::uint64_t mix(std::uint64_t a,std::uint64_t b){return a^(b+0x9e3779b97f4a7c15ULL+(a<<6)+(a>>2));}
std::uint64_t scalar(double value){std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));return bits;}
struct ImageFixture {
    std::size_t w,h;std::vector<std::uint16_t> depth;std::vector<std::uint8_t> rgb;
    std::vector<r::Pixel> support;r::Intrinsics k;
    ImageFixture(std::size_t width,std::size_t height):w(width),h(height),depth(w*h),rgb(w*h*3),k{w,h,570.*w/640,570.*w/640,(w-1)/2.,(h-1)/2.}{
        std::uint32_t seed=13;support.reserve(w*h);
        for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x){
            seed=1664525*seed+1013904223;const auto i=y*w+x;
            const bool target=x>w*3/8&&x<w*5/8&&y>h/6&&y<h*5/6;
            const bool side=x<w/5;
            depth[i]=static_cast<std::uint16_t>((target?1800:(side?2900:4000))+seed%21);
            if(seed%97==0)depth[i]=0;
            rgb[3*i]=target?220:(side?20:70);rgb[3*i+1]=target?30:(side?180:80);rgb[3*i+2]=target?20:110;
            if(target)support.push_back({x,y});
        }
    }
    r::DepthView<std::uint16_t> dv()const{return {depth.data(),depth.size(),w,h,.001};}
    r::RgbView rv()const{return {rgb.data(),rgb.size(),w,h};}
};
struct Runner {
    std::size_t iterations=101;std::ostream& out;std::string filter;
    template<class F>void run(const std::string& name,F fn,std::size_t batch=1){
        if(!filter.empty()&&name.find(filter)==std::string::npos)return;
        allocation_probe::calls=0;allocation_probe::bytes=0;allocation_probe::active=true;
        const auto cold_start=std::chrono::steady_clock::now();const auto cold_result=fn();
        const auto cold_end=std::chrono::steady_clock::now();allocation_probe::active=false;
        const auto cold_calls=allocation_probe::calls,cold_bytes=allocation_probe::bytes;
        const double cold_ms=std::chrono::duration<double,std::milli>(cold_end-cold_start).count();
        consumed=mix(consumed,cold_result);
        for(int i=1;i<5;++i)consumed=mix(consumed,fn());
        std::vector<double> ms;ms.reserve(iterations);std::size_t allocations=0,bytes=0;std::uint64_t digest=0;
        for(std::size_t i=0;i<iterations;++i){
            allocation_probe::calls=0;allocation_probe::bytes=0;allocation_probe::active=true;
            const auto start=std::chrono::steady_clock::now();
            for(std::size_t j=0;j<batch;++j){std::atomic_signal_fence(std::memory_order_seq_cst);digest=mix(digest,fn());std::atomic_signal_fence(std::memory_order_seq_cst);}
            const auto end=std::chrono::steady_clock::now();allocation_probe::active=false;
            allocations+=allocation_probe::calls;bytes+=allocation_probe::bytes;
            ms.push_back(std::chrono::duration<double,std::milli>(end-start).count()/batch);
        }
        consumed=mix(consumed,digest);std::sort(ms.begin(),ms.end());
        auto p=[&](double q){return ms[static_cast<std::size_t>(std::ceil(q*ms.size()))-1];};
        const double calls=static_cast<double>(iterations*batch);
        out<<name<<','<<iterations<<','<<batch<<','<<std::setprecision(9)<<p(.5)<<','<<p(.95)<<','<<p(.99)<<','<<ms.back()<<','
           <<allocations/calls<<','<<bytes/calls<<','<<cold_ms<<','<<cold_calls<<','<<cold_bytes<<','<<digest<<'\n';out.flush();
    }
};
int main(int argc,char** argv){
    try{
        std::size_t iterations=101;std::string filter;
        for(int i=1;i<argc;++i){
            const std::string argument=argv[i];
            if((argument=="--iterations"||argument=="--filter")&&i+1<argc){
                if(argument=="--iterations")iterations=static_cast<std::size_t>(std::stoul(argv[++i]));
                else filter=argv[++i];
            }else throw std::invalid_argument("usage: benchmark [--iterations N] [--filter substring]");
        }
        if(iterations<5||iterations>10000)throw std::invalid_argument("iterations must be 5..10000");
        std::cout<<"case,iterations,batch,p50_ms,p95_ms,p99_ms,max_ms,new_calls_per_op,new_bytes_per_op,cold_ms,cold_new_calls,cold_new_bytes,digest\n";
        Runner run{iterations,std::cout,filter};
        for(const auto extent:{std::pair<std::size_t,std::size_t>{640,480},{1280,720}}){
            ImageFixture im(extent.first,extent.second);const auto size=std::to_string(im.w)+"x"+std::to_string(im.h);
            run.run("depth_points_"+size,[&]{auto v=r::depth_to_points(im.dv(),im.k,{0,0,im.w,im.h});return mix(v.size(),scalar(v.back().point_m.x));});
            run.run("depth_regions_"+size,[&]{auto v=r::extract_depth_regions(im.dv(),im.k,{0,0,im.w,im.h});return mix(v.size(),scalar(v.front().mad_m));});
            run.run("depth_band_roi_"+size,[&]{auto v=r::estimate_depth_target(im.dv(),im.k,{im.w*3/8+1,im.h/5,im.w/4-2,im.h*3/5});return mix(v.support_count,scalar(v.depth_m.value_or(0)));});
            run.run("appearance_"+size,[&]{auto v=r::color_descriptor(im.rv(),im.support);return mix(v.size(),scalar(v[48]));});
            r::RgbdTracker tracker;std::size_t frame=0;
            run.run("rgbd_tracker_"+size,[&]{const double t=static_cast<double>(frame)/30;auto result=tracker.process(im.rv(),im.dv(),im.k,{}, {t,t,t,t},frame++==0?std::optional<r::Pixel>{{im.w/2,im.h/2}}:std::nullopt);
                return mix(mix(result.region_count,static_cast<std::uint64_t>(result.state)),scalar(result.observed_position_m?result.observed_position_m->z:0));});
        }
        {
            ImageFixture im(640,480);std::vector<float> floats(im.depth.size());
            for(std::size_t i=0;i<floats.size();++i)floats[i]=static_cast<float>(im.depth[i]*.001);
            r::DepthView<float> view(floats.data(),floats.size(),im.w,im.h,1.);
            r::RgbdTracker tracker;std::size_t frame=0;
            run.run("rgbd_tracker_float32_640x480",[&]{const double t=static_cast<double>(frame)/30;auto result=tracker.process(im.rv(),view,im.k,{}, {t,t,t,t},frame++==0?std::optional<r::Pixel>{{im.w/2,im.h/2}}:std::nullopt);
                return mix(result.region_count,scalar(result.observed_position_m?result.observed_position_m->z:0));});
            for(std::size_t y=0;y<im.h;++y)for(std::size_t x=0;x<im.w;++x){const auto i=y*im.w+x;
                const bool tile=(x/16+y/16)%2;im.depth[i]=tile?1900:2400;im.rgb[i*3]=tile?220:20;im.rgb[i*3+1]=tile?20:220;im.rgb[i*3+2]=30;}
            r::RgbdTracker fragmented;frame=0;
            run.run("rgbd_tracker_fragmented_640x480",[&]{const double t=static_cast<double>(frame)/30;auto result=fragmented.process(im.rv(),im.dv(),im.k,{}, {t,t,t,t},frame++==0?std::optional<r::Pixel>{{im.w/2,im.h/2}}:std::nullopt);
                return mix(result.region_count,static_cast<std::uint64_t>(result.state));});
        }
        std::vector<float> ranges;std::vector<n::Pose2> poses;
        for(std::size_t rays:{721,2000}){
            ranges.assign(rays,4.f);for(std::size_t i=0;i<rays;i+=71)ranges[i]=1.5f;
            n::LaserScanView scan{ranges.data(),rays,-n::pi,2*n::pi/(rays-1),.15,10,1,0};
            run.run("laser_grid_"+std::to_string(rays),[&]{auto v=n::laser_to_grid(scan,{},1);return mix(mix(v.usable_rays,v.obstacle_returns),static_cast<std::uint64_t>(v.grid->at(n::Point2{.5,.2})));});
            scan.time_increment_s=.1/(rays-1);poses=n::constant_twist_deskew(scan,{.2,.05,.1},1.1);
            run.run("laser_grid_deskew_"+std::to_string(rays),[&]{auto v=n::laser_to_grid(scan,{},1.1,{},poses);return mix(mix(v.usable_rays,scalar(v.obstacle_points.front().x)),static_cast<std::uint64_t>(v.grid->at(n::Point2{.5,.2})));});
        }
        n::LocalGrid grid;grid.fill(n::Cell::free);n::RobotFootprint robot{.25,.25,.20,.04};n::MotionLimits limits;
        std::vector<n::Point2> obstacles;
        for(std::size_t i=0;i<2000;++i){const double a=2*n::pi*i/2000;n::Point2 p{2.3*std::cos(a),2.3*std::sin(a)};obstacles.push_back(p);grid.mark_occupied(p);}
        // Benchmarks deliberately mix known free space, unknown space and blockers.
        auto mixed=grid;for(std::size_t i=0;i<180;++i)mixed.mark_occupied({1.+static_cast<double>(i%7)*.05,-.5+static_cast<double>(i/7)*.05});
        std::size_t pose_index=0;
        run.run("footprint_mixed",[&]{const double a=.013*static_cast<double>(pose_index++%100);auto v=n::check_footprint(mixed,{.8,.05,a},robot,.05);return static_cast<std::uint64_t>(v.state);},100);
        run.run("mark_robot_body",[&]{n::mark_robot_body(grid,robot);return static_cast<std::uint64_t>(grid.at(n::Point2{0,0}));},100);
        n::TrajectoryConfig tc;tc.obstacle_speed_bound_mps=0;
        run.run("trajectory_static",[&]{auto v=n::check_trajectory(grid,robot,limits,{.2,0,.1},{.22,.01,.15},tc);return mix(v.clear,scalar(v.end_pose.x));});
        tc.obstacle_speed_bound_mps=.5;
        run.run("trajectory_dynamic",[&]{auto v=n::check_trajectory(grid,robot,limits,{.2,0,.1},{.22,.01,.15},tc);return mix(v.clear,scalar(v.end_pose.x));});
        n::AvoidanceConfig ac;ac.trajectory.obstacle_speed_bound_mps=0;
        run.run("planner_static_2000_returns",[&]{auto v=n::plan_local_motion(grid,obstacles,robot,limits,{.2,0,0},{.4,0,0},{1.2,0},ac);return mix(mix(v.evaluated,v.rejected),scalar(v.command.x));});
        ac.trajectory.obstacle_speed_bound_mps=.5;
        run.run("planner_dynamic_2000_returns",[&]{auto v=n::plan_local_motion(grid,obstacles,robot,limits,{.2,0,0},{.4,0,0},{1.2,0},ac);return mix(mix(v.evaluated,v.rejected),scalar(v.command.x));});
        auto detour_grid=grid;std::vector<n::Point2> detour_obstacles;
        for(double y=-.1;y<=.1;y+=.025){detour_obstacles.push_back({.8,y});detour_grid.mark_occupied({.8,y});}
        ac.trajectory.obstacle_speed_bound_mps=0;ac.allow_detour=true;
        run.run("planner_static_detour",[&]{auto v=n::plan_local_motion(detour_grid,detour_obstacles,robot,limits,{}, {.4,0,0},{1.2,0},ac);return mix(v.found,scalar(v.command.y));});
        n::NavigationConfig nc;n::FollowNavigator nav(robot,limits,nc);std::size_t tick=0;n::LaserObservation observation;observation.grid=grid;observation.obstacle_points=obstacles;observation.usable=true;
        run.run("navigator_dynamic",[&]{const double t=.1*static_cast<double>(tick++);observation.first_stamp_s=t;observation.last_stamp_s=t;observation.reference_stamp_s=t;
            auto v=nav.step({{2.4,0},t,true},observation,{{.2,0,0},t},t);return mix(v.moving_allowed,scalar(v.command.x));});
        r::Motion3D motion;motion.initialize({0,0,2},0);std::size_t update=0;
        run.run("motion3d_update",[&]{const double t=.01*static_cast<double>(++update);bool ok=motion.update({t*.1,0,2},t);return static_cast<std::uint64_t>(ok);},100);
        std::vector<r::TrackCandidate> candidates;
        for(std::size_t i=0;i<100;++i){std::vector<double> d(128,.01);d[i%128]=1.;candidates.push_back({i,{.1*static_cast<double>(i),0,2},std::move(d)});}
        r::AssociationQuery query{{0,0,2},{.01,.01,.01},candidates.front().appearance};
        run.run("association_100x128",[&]{auto v=r::associate_target(candidates,query);return mix(static_cast<std::uint64_t>(v.state),v.candidate_index.value_or(999));},10);
        run.run("follow_control",[&]{auto v=n::follow_command({{2.4,.1},0,true},0,limits);return scalar(v.velocity.x);},100);
        std::cerr<<"checksum="<<consumed<<"\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
