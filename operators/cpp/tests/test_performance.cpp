#include <operator_lab/rgbd_tracker.hpp>
#include <operator_lab/follow_navigation.hpp>
#include <iostream>
#include <random>

namespace r=operator_lab::rgbd;
namespace n=operator_lab::navigation;
namespace {
std::size_t checks=0;
#define CHECK(x) do{++checks;if(!(x))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+" " #x);}while(false)
double sorted_median(std::vector<double> values){
    std::sort(values.begin(),values.end());const auto count=values.size();
    return count%2?values[count/2]:values[count/2-1]/2+values[count/2]/2;
}
template<class T>void verify_regions(const std::vector<r::DepthRegion>& regions,const r::DepthView<T>& depth){
    for(const auto& region:regions){
        std::vector<double> values;for(auto p:region.pixels)values.push_back(depth.at_m(p.u,p.v));
        const double median=sorted_median(values);
        for(auto& z:values)z=std::abs(z-median);
        CHECK(region.depth_m==median);
        if(region.mad_m!=sorted_median(values))std::cerr<<"MAD mismatch size="<<values.size()<<" median="<<std::hexfloat<<median<<" got="<<region.mad_m<<" expected="<<sorted_median(values)<<"\n";
        CHECK(region.mad_m==sorted_median(values));
        CHECK(region.point_m.z==depth.at_m(region.representative_pixel.u,region.representative_pixel.v));
    }
}
void statistics_and_workspace(){
    constexpr std::size_t w=80,h=64;std::vector<std::uint16_t> image(w*h);
    const r::Intrinsics k{w,h,80,80,40,32};std::mt19937 random(503);
    r::DepthRegionConfig config;config.min_depth_m=0;config.max_depth_m=1e308;config.absolute_jump_m=1e308;config.relative_jump=0;config.min_pixels=1;
    r::DepthRegionWorkspace workspace;
    for(int trial=0;trial<36;++trial){
        for(auto& z:image)z=static_cast<std::uint16_t>(1+random()%65535);
        if(trial%3==0)for(auto& z:image)z=static_cast<std::uint16_t>(100+random()%7);
        if(trial%2==0)image.back()=0; // Odd and even support counts.
        const double scale=trial%4==0?1e-280:(trial%4==1?1e280:.001);
        r::DepthView<std::uint16_t> view(image.data(),image.size(),w,h,scale);
        auto result=r::extract_depth_regions(view,k,{0,0,w,h},config,std::nullopt,{},&workspace);
        CHECK(result.size()==1);verify_regions(result,view);
        const auto saved=result.front().pixels;
        std::fill(image.begin(),image.end(),0);
        CHECK(r::extract_depth_regions(view,k,{0,0,w,h},config,std::nullopt,{},&workspace).empty());
        CHECK(result.front().pixels.size()==saved.size());
        for(std::size_t i=0;i<saved.size();++i)CHECK(result.front().pixels[i].u==saved[i].u&&result.front().pixels[i].v==saved[i].v);
    }
    // General floating depths use exact selection rather than raw-code histograms.
    for(int trial=0;trial<20;++trial){
        std::vector<float> values(11*13);
        for(auto& z:values)z=static_cast<float>(1.+(random()%3000)/1000.);
        r::DepthView<float> view(values.data(),values.size(),11,13,1.);
        auto result=r::extract_depth_regions(view,{11,13,11,13,5,6},{0,0,11,13},config,std::nullopt,{},&workspace);
        CHECK(result.size()==1);verify_regions(result,view);
    }
    // Large floating components exercise the fallback without histogram quantization.
    for(int trial=0;trial<4;++trial){
        std::vector<double> values(80*64);
        for(auto& z:values)z=trial%2?2.:(1.+(random()%300000)/100000.);
        r::DepthView<double> view(values.data(),values.size(),80,64,1.);
        verify_regions(r::extract_depth_regions(view,k,{0,0,80,64},config,std::nullopt,{},&workspace),view);
    }
    workspace.clear();CHECK(workspace.depth_m.empty()&&workspace.queue.empty()&&workspace.histogram.empty());
    std::fill(image.begin(),image.end(),2000);
    r::DepthView<std::uint16_t> view(image.data(),image.size(),w,h,.001);
    const auto connector=[](r::Pixel a,r::Pixel b){return (a.u<40)==(b.u<40);};
    auto direct=r::extract_depth_regions(view,k,{0,0,w,h},config,std::nullopt,connector,&workspace);
    std::function<bool(r::Pixel,r::Pixel)> indirect=connector;
    auto erased=r::extract_depth_regions(view,k,{0,0,w,h},config,std::nullopt,indirect);
    CHECK(direct.size()==2&&erased.size()==2);
    for(std::size_t i=0;i<direct.size();++i)CHECK(direct[i].pixels.size()==erased[i].pixels.size());
    // The pre-existing callback API also accepted mutable lambda targets.
    auto mutable_callback=[count=0](r::Pixel,r::Pixel)mutable{return ++count>0;};
    CHECK(r::extract_depth_regions(view,k,{0,0,w,h},config,std::nullopt,mutable_callback).size()==1);
    CHECK(r::extract_depth_regions(view,k,{0,0,w,h},config,std::nullopt,nullptr).size()==1);
    // Strided views preserve capacity checks and raw/metric pixel correspondence.
    std::vector<std::uint16_t> strided((w+7)*h,3000);
    r::DepthView<std::uint16_t> stride(strided.data(),strided.size(),w,h,.001,w+7);
    CHECK(stride.raw_at(79,63)==3000&&stride.scale_to_m()==.001);
    verify_regions(r::extract_depth_regions(stride,k,{0,0,w,h},config,std::nullopt,{},&workspace),stride);
}
void body_marking(){
    std::mt19937 random(77);
    for(int trial=0;trial<60;++trial){
        // Includes maps not centered on the robot and bodies partly outside maps.
        const double shift=static_cast<double>(trial%7-3)*.23;
        n::GridConfig config{-1+shift,-.7-shift,1+shift,.9-shift,.04};
        n::LocalGrid actual(config);auto expected=actual;
        n::RobotFootprint robot{.10+(random()%70)/100.,.1+(random()%40)/100.,.1+(random()%60)/100.,.03};
        actual.mark_occupied({.05,0});expected.mark_occupied({.05,0});
        n::mark_robot_body(actual,robot);
        const double half=config.resolution/2;
        for(std::size_t i=0;i<expected.size();++i){
            const auto p=expected.center(i);
            if(p.x-half>=-robot.back&&p.x+half<=robot.front&&p.y-half>=-robot.half_width&&p.y+half<=robot.half_width)expected.mark_free(p);
        }
        for(std::size_t i=0;i<actual.size();++i)CHECK(actual.at(i)==expected.at(i));
    }
}
void path_storage(){
    n::LocalGrid grid;grid.fill(n::Cell::free);n::RobotFootprint robot{.25,.25,.2,.03};n::MotionLimits limits;
    for(int mode=0;mode<4;++mode){
        if(mode==1)grid.mark_occupied({.8,.05});
        if(mode==2)grid.fill(n::Cell::unknown);
        if(mode==3)grid.fill(n::Cell::free);
        for(int i=0;i<20;++i){
            n::TrajectoryConfig config;config.obstacle_speed_bound_mps=i%2?.5:0;
            const n::Velocity2 current{.1+i*.005,.02,.1},command=current;
            auto stored=n::check_trajectory(grid,robot,limits,current,command,config,.02,false,true);
            auto lean=n::check_trajectory(grid,robot,limits,current,command,config,.02,false,false);
            CHECK(stored.clear==lean.clear&&stored.collision==lean.collision);
            CHECK(stored.time_s==lean.time_s&&stored.end_pose.x==lean.end_pose.x&&stored.end_pose.y==lean.end_pose.y&&stored.end_pose.yaw==lean.end_pose.yaw);
            CHECK(stored.blocking_cell.has_value()==lean.blocking_cell.has_value());
            if(stored.blocking_cell)CHECK(stored.blocking_cell->x==lean.blocking_cell->x&&stored.blocking_cell->y==lean.blocking_cell->y);
            CHECK(!stored.path.empty()&&lean.path.empty());
        }
    }
}
}
int main(){try{statistics_and_workspace();body_marking();path_storage();std::cout<<"PASS: "<<checks<<" performance-regression checks (exact statistics, workspace, body marking, path storage)\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
