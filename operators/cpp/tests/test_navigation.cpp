#include <operator_lab/tracking_navigation.hpp>
#include <operator_lab/laser_scan_io.hpp>
#include <chrono>
#include <iostream>
#include <random>

using namespace operator_lab::navigation;
namespace {
std::size_t checks=0;
#define CHECK(x) do{++checks;if(!(x))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+" " #x);}while(false)
template<class F>void throws(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}CHECK(caught);}
bool near(double a,double b,double e=1e-8){return std::abs(a-b)<=e;}
RobotFootprint robot(){return {.25,.25,.2,.03};}
LocalGrid free_grid(){LocalGrid grid;grid.fill(Cell::free);return grid;}
void geometry_tests(){
    const auto quarter=integrate({}, {1,0,1},pi/2);
    CHECK(near(quarter.x,1)&&near(quarter.y,1)&&near(quarter.yaw,pi/2));
    const auto lateral=integrate({}, {0,1,1},pi/2);CHECK(near(lateral.x,-1)&&near(lateral.y,1));
    CHECK(near(integrate({}, {1,2,0},.2).y,.4));
    const auto tiny=integrate({}, {1,0,1e-12},1);CHECK(near(tiny.x,1)&&std::abs(tiny.y)<1e-10);
    const auto p=transform_point(transform_point({.5,-.2},{.2,.3,.4}),inverse({.2,.3,.4}));
    CHECK(near(p.x,.5)&&near(p.y,-.2));
    throws([]{validate(RobotFootprint{});});throws([]{MotionLimits l;l.brake_linear=0;validate(l);});
    MotionLimits limits;auto v=bound_velocity({1,1,5},limits);
    CHECK(speed(v)<=limits.max_linear+1e-9&&std::abs(v.yaw)<=limits.max_yaw);
    const auto smooth=approach_velocity({}, {.3,.3,1},limits,.1);
    CHECK(speed(smooth)<=.05+1e-9&&near(smooth.yaw,.1));
    CHECK(stopped(brake_velocity({.3,.1,.4},limits,10)));
}
void laser_tests(){
    std::vector<float> ranges(721,5);
    LaserScanView scan{ranges.data(),ranges.size(),-pi,2*pi/720,.02,5,1,0};
    auto laser=laser_to_grid(scan,{},1);CHECK(laser.usable&&laser.grid&&laser.usable_rays==721);
    CHECK(laser.grid->at(Point2{.5,0})==Cell::free);
    CHECK(laser_to_grid(scan,{},1.3).reason=="stale_scan");
    CHECK(laser_to_grid(scan,{},.9).reason=="future_scan");
    std::fill(ranges.begin(),ranges.end(),std::numeric_limits<float>::infinity());
    CHECK(!laser_to_grid(scan,{},1).usable);
    LaserConfig c;c.positive_infinity_is_clear=true;
    laser=laser_to_grid(scan,{},1,c);CHECK(laser.usable&&laser.obstacle_returns==0);
    std::fill(ranges.begin(),ranges.end(),std::numeric_limits<float>::quiet_NaN());
    CHECK(!laser_to_grid(scan,{},1,c).usable);
    std::fill(ranges.begin(),ranges.end(),5);ranges[360]=.5f;
    laser=laser_to_grid(scan,{},1);CHECK(laser.grid->at(Point2{.5,0})==Cell::occupied);
    CHECK(laser.grid->at(Point2{.7,0})!=Cell::occupied);
    scan.time_increment_s=.1/720;
    CHECK(laser_to_grid(scan,{},1.1).reason=="deskew_required");
    const auto poses=constant_twist_deskew(scan,{1,0,0},1.1);
    CHECK(near(poses.front().x,-.1)&&near(poses.back().x,0));
    laser=laser_to_grid(scan,{},1.1,{},poses);CHECK(laser.usable);
    CHECK(near(laser.obstacle_points[360].x,.45,1e-7));
    throws([&]{laser_to_grid(scan,{},1.1,{},std::vector<Pose2>(2));});
    throws([&]{constant_twist_deskew(scan,{},1.05);});
    scan.time_increment_s=0;
    laser=laser_to_grid(scan,{.1,0,pi/2},1);CHECK(laser.usable);
    CHECK(std::any_of(laser.obstacle_points.begin(),laser.obstacle_points.end(),[](Point2 p){return near(p.x,.1)&&near(p.y,.5);}));
    // Minimum range is a blind zone, not known-free space.
    scan.range_min=.4;laser=laser_to_grid(scan,{},1);
    CHECK(laser.grid->at(Point2{.1,0})==Cell::unknown);
    // Negative angle increments and partial FOV are valid; unknown side space persists.
    std::vector<float> partial(181,3);LaserScanView front{partial.data(),partial.size(),pi/2,-pi/180,.02,3,0,0};
    laser=laser_to_grid(front,{},0);CHECK(laser.usable&&laser.grid->at(Point2{-1,0})==Cell::unknown);
    throws([&]{LaserScanView bad=scan;bad.angle_increment=0;laser_to_grid(bad,{},1);});
}
void footprint_tests(){
    auto grid=free_grid();grid.mark_occupied({.19,.19});RobotFootprint thin{.3,.3,.1,0};
    CHECK(check_footprint(grid,{},thin).state==CollisionState::clear);
    CHECK(check_footprint(grid,{0,0,pi/4},thin).state==CollisionState::occupied);
    CHECK(check_footprint(grid,{2.95,0,0},robot()).state==CollisionState::unknown);
    LocalGrid unknown;mark_robot_body(unknown,robot());
    CHECK(unknown.at(Point2{0,0})==Cell::free);
    CHECK(check_footprint(unknown,{},robot()).state==CollisionState::unknown);
    unknown.mark_occupied({.1,0});mark_robot_body(unknown,robot());CHECK(unknown.at(Point2{.1,0})==Cell::occupied);
    // Independent interval-overlap reference for axis-aligned footprints.
    std::mt19937 random(42);
    for(int i=0;i<500;++i){
        Point2 point{(static_cast<int>(random()%200)-100)/100.,(static_cast<int>(random()%200)-100)/100.};
        const double half=.025;
        const bool expected=point.x+half>=-.25-.03&&point.x-half<=.25+.03&&
                            point.y+half>=-.2-.03&&point.y-half<=.2+.03;
        CHECK(footprint_intersects_cell(point,half,{},robot())==expected);
    }
}
void trajectory_tests(){
    auto grid=free_grid();MotionLimits limits;TrajectoryConfig c;
    const auto stop=check_trajectory(grid,robot(),limits,{.4,0,0},{},c,0,true);
    CHECK(stop.clear);
    const double analytic=.4*c.reaction_s+.4*.4/(2*limits.brake_linear);
    CHECK(stop.end_pose.x>=analytic-1e-8&&stop.end_pose.x<=analytic+.4*c.time_step_s+1e-8);
    c.reaction_s=.3;
    const auto delayed=check_trajectory(grid,robot(),limits,{.4,0,0},{},c,0,true);
    CHECK(near(delayed.end_pose.x-stop.end_pose.x,.08,1e-6));
    c={};grid.mark_occupied({.45,0});
    CHECK(!check_trajectory(grid,robot(),limits,{.4,0,0},{},c,0,true).clear);
    throws([&]{check_trajectory(grid,robot(),limits,{}, {.4,0,0},c);});
    grid=free_grid();for(double x=-.1;x<=.1;x+=.025)grid.mark_occupied({x,.6});
    CHECK(!check_trajectory(grid,robot(),limits,{0,.3,0},{0,.3,0},c).clear);
    grid=free_grid();grid.mark_occupied({.22,.15});
    CHECK(!check_trajectory(grid,{.3,.3,.1,0},limits,{0,0,1},{},c,0,true).clear);
    // A thin wall must not be skipped at high speed or a coarse nominal time step.
    grid=free_grid();for(double y=-1;y<=1;y+=.025)grid.mark_occupied({.8,y});
    limits.max_linear=3;limits.accel_linear=10;limits.brake_linear=10;c.time_step_s=.2;c.reaction_s=0;
    CHECK(!check_trajectory(grid,robot(),limits,{2,0,0},{2,0,0},c).clear);
    LocalGrid unknown;CHECK(!check_trajectory(unknown,robot(),MotionLimits{}, {},{},TrajectoryConfig{}).clear);
    grid=free_grid();grid.mark_occupied({1.2,0});limits={};c={};c.obstacle_speed_bound_mps=0;
    CHECK(check_trajectory(grid,robot(),limits,{.2,0,0},{.2,0,0},c).clear);
    c.obstacle_speed_bound_mps=.7;
    CHECK(!check_trajectory(grid,robot(),limits,{.2,0,0},{.2,0,0},c).clear);
}
void control_and_planner_tests(){
    MotionLimits limits;auto f=follow_command({{2.5,0},0,true},0,limits);CHECK(f.target_usable&&f.velocity.x>0);
    CHECK(stopped(follow_command({{1.2,0},0,true},0,limits).velocity));
    CHECK(stopped(follow_command({{.5,0},0,true},0,limits).velocity));
    f=follow_command({{0,2.5},0,true},0,limits);CHECK(f.velocity.x==0&&f.velocity.yaw>0);
    CHECK(!follow_command({{2,0},0,false},0,limits).target_usable);
    CHECK(!follow_command({{2,0},0,true},.3,limits).target_usable);
    CHECK(!follow_command({{9,0},0,true},0,limits).target_usable);
    CHECK(!follow_command({{1e308,1e308},0,true},0,limits).target_usable);
    auto grid=free_grid();std::vector<Point2> points;
    for(double y=-.1;y<=.1;y+=.025){points.push_back({.8,y});grid.mark_occupied({.8,y});}
    AvoidanceConfig c;
    c.trajectory.obstacle_speed_bound_mps=0; // Explicit static-obstacle fixture.
    auto plan=plan_local_motion(grid,points,robot(),limits,{}, {.4,0,0},{1.5,0},c);
    CHECK(!plan.found&&plan.reason=="direct_path_blocked");
    c.allow_detour=true;plan=plan_local_motion(grid,points,robot(),limits,{}, {.4,0,0},{1.5,0},c);
    CHECK(plan.found&&plan.goal.detour&&std::abs(plan.command.y)>.001);
    CHECK(plan.trajectory.clear&&speed(plan.command)<=limits.accel_linear*c.trajectory.control_dt_s+1e-8);
    CHECK(check_trajectory(grid,robot(),limits,{},plan.command,c.trajectory).clear);
    points.push_back({.8,5});points.push_back({.8,-5});
    CHECK(plan_local_motion(grid,points,robot(),limits,{}, {.4,0,0},{1.5,0},c).found);
    // Fully blocked passage cannot be declared a bypass.
    grid=free_grid();points.clear();
    for(double y=-3;y<3;y+=.025){points.push_back({.7,y});grid.mark_occupied({.7,y});}
    plan=plan_local_motion(grid,points,robot(),limits,{}, {.4,0,0},{1.5,0},c);
    CHECK(!plan.found);
}
LaserObservation synthetic_scan(double stamp,const LocalGrid& grid){
    LaserObservation scan;scan.usable=true;scan.grid=grid;scan.first_stamp_s=stamp;scan.last_stamp_s=stamp;scan.reference_stamp_s=stamp;
    for(std::size_t i=0;i<grid.size();++i)if(grid.at(i)==Cell::occupied)scan.obstacle_points.push_back(grid.center(i));return scan;
}
void arbiter_tests(){
    auto grid=free_grid();FollowNavigator navigator(robot(),MotionLimits{});
    NavigationResult result;
    for(int i=0;i<3;++i){const double t=i*.1;result=navigator.step({{2.5,0},t,true},synthetic_scan(t,grid),{{},t},t);}
    CHECK(result.moving_allowed&&result.command.x>0);
    result=navigator.step({{2.5,0},.3,true},synthetic_scan(.3,grid),{{},.3},.3,true);
    CHECK(!result.moving_allowed&&stopped(result.command)&&result.reason=="emergency_stop");
    for(int i=0;i<4;++i){const double t=.31+i*.01;auto laser=synthetic_scan(.3,grid);laser.reference_stamp_s=t;
        result=navigator.step({{2.5,0},t,true},laser,{{},t},t);CHECK(!result.moving_allowed);}
    result=navigator.step({{2.5,0},.4,true},synthetic_scan(.4,grid),{{},.4},.4);CHECK(!result.moving_allowed);
    result=navigator.step({{2.5,0},.5,true},synthetic_scan(.5,grid),{{},.5},.5);CHECK(result.moving_allowed);
    result=navigator.step({{2.5,0},.6,false},synthetic_scan(.6,grid),{{},.6},.6);
    CHECK(!result.moving_allowed&&stopped(result.command));
    auto stale=synthetic_scan(.3,grid);stale.reference_stamp_s=.7;
    CHECK(navigator.step({{2.5,0},.7,true},stale,{{},.7},.7).reason=="laser_stale_or_future");
    CHECK(navigator.step({{2.5,0},.8,true},synthetic_scan(.8,grid),{{},.4},.8).reason=="invalid_or_stale_velocity");
    CHECK(navigator.step({{2.5,0},.9,true},synthetic_scan(.9,grid),{{},.9},.95).reason=="grid_reference_outdated");
    CHECK(navigator.step({{2.5,0},1,true},synthetic_scan(.55,grid),{{},1},1).reason=="laser_stale_or_future");
    // Close pedestrian occupies the nominal path: command zero and clear-count reset.
    auto blocked=grid;blocked.mark_occupied({.6,0});
    result=navigator.step({{2.5,0},1.1,true},synthetic_scan(1.1,blocked),{{},1.1},1.1);
    CHECK(!result.moving_allowed&&stopped(result.command));
    result=navigator.step({{2.5,0},1.2,true},synthetic_scan(1.2,grid),{{},1.2},1.2);CHECK(!result.moving_allowed);
    result=navigator.step({{2.5,0},1.3,true},synthetic_scan(1.3,grid),{{},1.3},1.3);CHECK(!result.moving_allowed);
    result=navigator.step({{2.5,0},1.4,true},synthetic_scan(1.4,grid),{{},1.4},1.4);CHECK(result.moving_allowed);
    CHECK(navigator.step({{2,0},0,true},synthetic_scan(0,grid),{{},0},0).reason=="control_clock_rewind");
    CHECK(navigator.step({{NAN,0},.1,true},synthetic_scan(.1,grid),{{},.1},.1).reason=="invalid_target");
    CHECK(navigator.step({{2,0},.2,true},synthetic_scan(.2,grid),{{},.2},NAN).reason=="nonfinite_control_time");
}
void bridge_tests(){
    TrackingTargetCache cache;operator_lab::rgbd::TrackerResult result;
    result.fresh_observation=true;result.confirmed=true;result.observed_position_m=operator_lab::rgbd::Point3{2.5,.2,1};
    cache.observe(result,1);
    operator_lab::rgbd::RigidTransform base_from_fixed;base_from_fixed.translation_m.x=-.5;
    auto target=cache.target(base_from_fixed);CHECK(near(target.point_base.x,2)&&near(target.stamp_s,1)&&target.confirmed);
    CHECK(!follow_command(target,1.3,MotionLimits{}).target_usable);
    result.fresh_observation=false;result.confirmed=false;cache.observe(result,1.4);
    CHECK(!cache.target(base_from_fixed).confirmed);
    throws([&]{cache.observe(result,1.4);});
    auto obstacles=height_obstacles({{.5,0,.1},{1,0,.8},{2,0,0},{2,0,2},{NAN,0,1}});
    CHECK(obstacles.size()==2);
    HeightObstacleConfig heights;heights.floor_height_m=-.5;
    CHECK(height_obstacles({{.5,0,-.4}},heights).size()==1);
    auto laser=synthetic_scan(1,free_grid());laser=add_obstacles(laser,obstacles,1,1);
    CHECK(laser.usable&&laser.grid->at(Point2{.5,0})==Cell::occupied);
    CHECK(!add_obstacles(laser,obstacles,1,1.2).usable);
    CHECK(!add_obstacles(laser,obstacles,.5,1).usable);
}
void scan_io_tests(){
    const auto path=std::filesystem::temp_directory_path()/("operator-laser-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".scan");
    std::vector<float> ranges{1,2,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),.123456789f};
    LaserScanView scan{ranges.data(),ranges.size(),-.5,.2,.02,5,1,.001};
    write_laser_scan(path,scan);const auto loaded=read_laser_scan(path);
    CHECK(loaded.ranges.size()==ranges.size()&&loaded.first_stamp_s==1&&loaded.time_increment_s==.001);
    CHECK(loaded.ranges[5]==ranges[5]&&std::isnan(loaded.ranges[2])&&std::isinf(loaded.ranges[3])&&loaded.ranges[4]<0);
    {std::ofstream file(path);file<<"OPERATOR_LASER_V1\n2 0 .1 .02 5 0 0\n1\n";}
    throws([&]{read_laser_scan(path);});
    {std::ofstream file(path);file<<"OPERATOR_LASER_V1\n2 0 .1 .02 5 0 0\n1 2 extra\n";}
    throws([&]{read_laser_scan(path);});
    std::filesystem::remove(path);
}
}
int main(){try{geometry_tests();laser_tests();footprint_tests();trajectory_tests();control_and_planner_tests();arbiter_tests();bridge_tests();scan_io_tests();
std::cout<<"PASS: "<<checks<<" C++ laser/control/collision/local-avoidance checks\n";}
catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
