#include <operator_lab/rgbd_tracker.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>

using namespace operator_lab::rgbd;
namespace {
std::size_t checks = 0;
#define CHECK(expr) do { ++checks; if (!(expr)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " #expr); } while (false)
template<class F> void throws(F f) { bool caught=false; try { f(); } catch (const std::exception&) { caught=true; } CHECK(caught); }
bool near(double a, double b, double tolerance=1e-8) { return std::abs(a-b)<=tolerance; }
struct Scene {
    std::size_t w=40,h=20;
    std::vector<std::uint16_t> depth=std::vector<std::uint16_t>(w*h,0);
    std::vector<std::uint8_t> rgb=std::vector<std::uint8_t>(w*h*3,0);
    Intrinsics k{w,h,40,40,20,10};
    void box(std::size_t x,std::uint16_t z,std::array<std::uint8_t,3> color) {
        for(std::size_t y=5;y<13;++y) for(std::size_t u=x;u<x+4;++u) {
            depth[y*w+u]=z;
            for(std::size_t c=0;c<3;++c) rgb[(y*w+u)*3+c]=color[c];
        }
    }
    RgbView colors() const { return {rgb.data(),rgb.size(),w,h}; }
    DepthView<std::uint16_t> depths() const { return {depth.data(),depth.size(),w,h,.001}; }
};
void regions_test() {
    Scene s; s.box(5,2000,{255,0,0}); s.box(25,2000,{0,0,255});
    auto regions=extract_depth_regions(s.depths(),s.k,{0,0,s.w,s.h});
    CHECK(regions.size()==2);
    for(const auto& r:regions) {
        CHECK(r.pixels.size()==32 && near(r.depth_m,2) && near(r.mad_m,0));
        CHECK(std::any_of(r.pixels.begin(),r.pixels.end(),[&](Pixel p){return p.u==r.representative_pixel.u&&p.v==r.representative_pixel.v;}));
        CHECK(near(r.point_m.z,s.depths().at_m(r.representative_pixel.u,r.representative_pixel.v)));
        CHECK(r.representative_pixel.u<9 || r.representative_pixel.u>=25);
    }
    std::vector<std::uint8_t> mask(s.w*s.h,0);
    for(std::size_t y=0;y<s.h;++y) for(std::size_t x=20;x<s.w;++x) mask[y*s.w+x]=1;
    regions=extract_depth_regions(s.depths(),s.k,{0,0,s.w,s.h},{},MaskView(mask.data(),mask.size(),s.w,s.h));
    CHECK(regions.size()==1 && regions[0].bounds.x==25);
    regions=extract_depth_regions(s.depths(),s.k,{5,5,4,8});
    CHECK(regions.size()==1 && regions[0].bounds.x==5 && regions[0].bounds.y==5);
    throws([&]{MaskView(nullptr,10,2,2);});
    throws([&]{MaskView(mask.data(),mask.size(),s.w,s.h,1);});
    throws([&]{extract_depth_regions(s.depths(),s.k,{39,0,2,20});});
    DepthRegionConfig bad; bad.connectivity=6;
    throws([&]{extract_depth_regions(s.depths(),s.k,{0,0,s.w,s.h},bad);});

    // Independent flood-fill reference for randomized binary support. Compare the
    // partition by pair membership, so ordering of labels is not assumed.
    std::mt19937 random(19);
    for(int trial=0;trial<40;++trial) {
        const std::size_t w=9,h=7;
        std::vector<float> depth(w*h);
        for(auto& z:depth) z=random()%3==0?0.f:2.f;
        std::vector<int> labels(w*h,-1); int label=0;
        for(std::size_t root=0;root<depth.size();++root) {
            if(depth[root]==0 || labels[root]>=0) continue;
            std::queue<std::size_t> q; q.push(root); labels[root]=label;
            while(!q.empty()) {
                const auto p=q.front();q.pop();
                for(std::size_t n=0;n<depth.size();++n) {
                    const auto dx=static_cast<int>(p%w)-static_cast<int>(n%w);
                    const auto dy=static_cast<int>(p/w)-static_cast<int>(n/w);
                    if(std::abs(dx)+std::abs(dy)==1 && depth[n]!=0 && labels[n]<0) { labels[n]=label;q.push(n); }
                }
            }
            ++label;
        }
        DepthRegionConfig c;c.min_pixels=1;c.absolute_jump_m=0;c.relative_jump=0;
        auto found=extract_depth_regions(DepthView<float>(depth.data(),depth.size(),w,h,1),{w,h,20,20,4,3},{0,0,w,h},c);
        CHECK(found.size()==static_cast<std::size_t>(label));
        for(const auto& region:found) {
            const auto first=region.pixels.front(); const int expected=labels[first.v*w+first.u];
            CHECK(region.pixels.size()==static_cast<std::size_t>(std::count(labels.begin(),labels.end(),expected)));
            for(Pixel pixel:region.pixels) CHECK(labels[pixel.v*w+pixel.u]==expected);
        }
    }
    std::vector<float> diagonal{2,0,0,2};DepthRegionConfig c;c.min_pixels=1;
    auto diagonal_view=DepthView<float>(diagonal.data(),4,2,2,1);
    CHECK(extract_depth_regions(diagonal_view,{2,2,1,1,0,0},{0,0,2,2},c).size()==2);
    c.connectivity=8;
    CHECK(extract_depth_regions(diagonal_view,{2,2,1,1,0,0},{0,0,2,2},c).size()==1);
}
void motion_test() {
    Motion3D filter;CHECK(!filter.predict(0)); filter.initialize({0,0,2},0);
    CHECK(near(filter.predict(0)->position_m.z,2));
    const auto a=filter.predict(.1),b=filter.predict(.5);
    CHECK(b->variance_m2[0]>a->variance_m2[0]);
    CHECK(!filter.predict(.61));
    CHECK(!filter.update({0,0,2},.61));
    CHECK(filter.predict(.1).has_value()); // An expired rejected update did not mutate state.
    throws([&]{filter.predict(-1);});throws([&]{filter.update({0,0,2},0);});
    throws([&]{filter.update({NAN,0,2},.1);});
    for(double dt:{.02,.05,.1}) {
        Motion3D moving;moving.initialize({0,0,2},0);
        for(int frame=1;frame<100;++frame) {
            const double time=frame*dt;
            CHECK(moving.update({.4*time,-.2*time,2},time));
        }
        const double time=99*dt;
        const auto p=moving.predict(time+.2);
        CHECK(p && near(p->position_m.x,.4*(time+.2),.02));
        CHECK(near(p->velocity_mps.x,.4,.02));
        const auto before=*moving.predict(time+.1);
        CHECK(!moving.update({100,100,100},time+.1));
        const auto after=*moving.predict(time+.1);
        CHECK(near(before.position_m.x,after.position_m.x));
        CHECK(near(before.variance_m2[0],after.variance_m2[0]));
        for(int repeat=0;repeat<5;++repeat) CHECK(moving.predict(time+.5).has_value());
        CHECK(!moving.predict(time+.7));
    }
    Motion3D long_run;long_run.initialize({0,0,2},0);
    for(int frame=1;frame<2000;++frame) CHECK(long_run.update({0,0,2},frame*.01));
    CHECK(long_run.predict(20)->variance_m2[0]>=0);
}
void association_test() {
    AssociationQuery q{{0,0,2},{.01,.01,.01},{1,0,0}};
    std::vector<TrackCandidate> candidates{{1,{.01,0,2},{0,0,1}},{2,{.2,0,2},{1,0,0}}};
    auto result=associate_target(candidates,q);
    CHECK(result.state==AssociationState::matched && *result.candidate_index==1);
    candidates={{1,{-.1,0,2},{1,0,0}},{2,{.1,0,2},{1,0,0}}};
    CHECK(associate_target(candidates,q).state==AssociationState::ambiguous);
    candidates={{1,{0,0,2},{0,0,1}}};
    CHECK(associate_target(candidates,q).state==AssociationState::no_match);
    candidates={{1,{NAN,0,2},{1,0,0}},{2,{0,0,2},{0,0,0}}};
    CHECK(associate_target(candidates,q).state==AssociationState::no_match);
    q.reference_appearance={0,0,0};throws([&]{associate_target({},q);});
    Scene s;s.box(5,2000,{255,0,0});
    std::vector<Pixel> pixels{{5,5},{6,5}};
    const auto descriptor=color_descriptor(s.colors(),pixels);
    CHECK(near(descriptor[48],1));
    CHECK(near(std::accumulate(descriptor.begin(),descriptor.end(),0.),1));
    std::vector<std::uint8_t> padded{0,0,255,7,7,7,0,0,255};
    RgbView bgr(padded.data(),padded.size(),1,2,6,ChannelOrder::bgr);
    CHECK(color_descriptor(bgr,{{0,0},{0,1}})==descriptor);
    throws([&]{color_descriptor(s.colors(),{});});
    throws([&]{RgbView(padded.data(),8,1,2,6);});
}
void tracker_test() {
    RgbdTracker tracker;
    auto run=[&](Scene& s,double stamp,std::optional<Pixel> seed=std::nullopt,RigidTransform pose=RigidTransform{}) {
        return tracker.process(s.colors(),s.depths(),s.k,pose,{stamp,stamp,stamp,stamp},seed);
    };
    Scene first;first.box(8,2000,{255,0,0});first.box(26,2000,{0,0,255});
    CHECK(run(first,0).state==TrackState::waiting_selection);
    auto result=run(first,.1,Pixel{9,8});CHECK(result.fresh_observation && !result.confirmed);
    result=run(first,.2);CHECK(result.state==TrackState::confirming);
    result=run(first,.3);CHECK(result.confirmed);
    Scene touching;touching.box(8,2000,{255,0,0});touching.box(10,2000,{0,0,255});
    RgbdTracker touching_tracker;
    touching_tracker.process(first.colors(),first.depths(),first.k,{}, {0,0,0,0},Pixel{9,8});
    const auto touching_result=touching_tracker.process(touching.colors(),touching.depths(),touching.k,{}, {.1,.1,.1,.1});
    CHECK(touching_result.fresh_observation && touching_result.bounds->x==8 && touching_result.bounds->width==2);
    const auto pixel=*touching_result.representative_pixel;
    CHECK(touching.colors().at(pixel.u,pixel.v)[0]==255);
    Scene crossing;crossing.box(8,1200,{0,0,255});crossing.box(13,2000,{255,0,0});
    result=run(crossing,.4);CHECK(result.fresh_observation && result.bounds->x==13);
    Scene hidden;hidden.box(13,1200,{0,0,255});
    result=run(hidden,.5);CHECK(result.state==TrackState::occluded && !result.confirmed && !result.observed_position_m);
    CHECK(tracker.poll(.51).state==TrackState::occluded);
    result=run(crossing,.6);CHECK(result.fresh_observation && !result.confirmed);
    run(crossing,.7);result=run(crossing,.8);CHECK(result.confirmed);
    result=tracker.poll(1.01);CHECK(!result.confirmed);
    CHECK(tracker.poll(1.5).state==TrackState::lost);
    result=run(crossing,1.6);CHECK(result.state==TrackState::lost && !result.fresh_observation);
    result=run(crossing,1.7,Pixel{14,8});CHECK(result.fresh_observation && !result.confirmed);
    const auto bad=tracker.process(crossing.colors(),crossing.depths(),crossing.k,{},
                                   {1.8,1.7,1.7,1.8});
    CHECK(bad.state==TrackState::rejected_frame && bad.reason=="rgb_depth_skew");
    result=tracker.process(crossing.colors(),crossing.depths(),crossing.k,{}, {1.7,1.7,1.7,1.85});
    CHECK(result.state==TrackState::rejected_frame && result.reason=="out_of_order_frame");
    throws([&]{tracker.poll(1.);});
    CHECK(run(crossing,0).state==TrackState::waiting_selection);
}
void ego_motion_test() {
    TrackerConfig config;config.evidence.max_speed_mps=.1;config.evidence.position_slack_m=.01;
    RgbdTracker tracker(config);
    // Camera translates 0.05 m/frame; the observed region shifts one pixel left,
    // while its transformed representative remains fixed in the world.
    for(int frame=0;frame<8;++frame) {
        Scene s;s.box(static_cast<std::size_t>(15-frame),2000,{255,0,0});
        RigidTransform pose;pose.translation_m.x=frame*.05;
        const double time=frame*.1;
        auto r=tracker.process(s.colors(),s.depths(),s.k,pose,{time,time,time,time},
                              frame==0?std::optional<Pixel>(Pixel{16,8}):std::nullopt);
        CHECK(r.fresh_observation);
        CHECK(r.confirmed==(frame>=2));
    }
}
}
int main() {
    try { regions_test();motion_test();association_test();tracker_test();ego_motion_test();
        std::cout<<"PASS: "<<checks<<" C++ spatial/association/motion/tracker checks\n";
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n';return 1; }
}
