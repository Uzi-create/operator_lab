#include <operator_lab/tracking_navigation.hpp>
#include <operator_lab/laser_scan_io.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace operator_lab::navigation;
namespace fs=std::filesystem;
namespace {
struct Box {double xmin,ymin,xmax,ymax;};
struct Scenario {std::string name;bool detour=false;int steps=100;};
std::vector<Box> obstacles(const Scenario& scene,double time){
    if(scene.name=="static_detour")return {{.8,-.15,1.1,.15}};
    if(scene.name=="blocked_passage")return {{.8,-3,1.,3}};
    if(scene.name=="crossing_person"&&time>=1.5&&time<=6.5){
        const double y=-1.+(time-1.5)*.4;return {{.85,y-.12,1.09,y+.12}};
    }
    return {};
}
double ray_box(Point2 origin,Point2 direction,Box box){
    double low=0,high=5;
    for(int axis=0;axis<2;++axis){
        const double o=axis?origin.y:origin.x,d=axis?direction.y:direction.x;
        const double minimum=axis?box.ymin:box.xmin,maximum=axis?box.ymax:box.xmax;
        if(std::abs(d)<1e-12){if(o<minimum||o>maximum)return 5;continue;}
        double a=(minimum-o)/d,b=(maximum-o)/d;if(a>b)std::swap(a,b);
        low=std::max(low,a);high=std::min(high,b);if(low>high)return 5;
    }
    return low<=high?low:5;
}
std::vector<float> scan_ranges(Pose2 robot,const std::vector<Box>& boxes){
    std::vector<float> ranges(721,5);
    for(std::size_t i=0;i<ranges.size();++i){
        const double angle=robot.yaw-pi+static_cast<double>(i)*2*pi/720;
        const Point2 direction{std::cos(angle),std::sin(angle)};
        double range=5;for(auto box:boxes)range=std::min(range,ray_box({robot.x,robot.y},direction,box));
        ranges[i]=range>=.02?static_cast<float>(range):std::numeric_limits<float>::quiet_NaN();
    }
    return ranges;
}
std::array<Point2,4> robot_corners(Pose2 pose,RobotFootprint robot){
    return {transform_point({robot.front,robot.half_width},pose),transform_point({-robot.back,robot.half_width},pose),
            transform_point({-robot.back,-robot.half_width},pose),transform_point({robot.front,-robot.half_width},pose)};
}
double point_segment(Point2 p,Point2 a,Point2 b){
    const double dx=b.x-a.x,dy=b.y-a.y,length=dx*dx+dy*dy;
    const double t=length>0?std::clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/length,0.,1.):0;
    return distance(p,{a.x+t*dx,a.y+t*dy});
}
double physical_clearance(Pose2 pose,RobotFootprint robot,Box box){
    // Exact rectangle SAT for overlap, followed by vertex-edge distances. This
    // oracle uses the continuous world boxes, independently of the scan/grid.
    const Point2 center{(box.xmin+box.xmax)/2,(box.ymin+box.ymax)/2};
    const double hx=(box.xmax-box.xmin)/2,hy=(box.ymax-box.ymin)/2;
    const double c=std::cos(pose.yaw),s=std::sin(pose.yaw),shift=(robot.front-robot.back)/2;
    const Point2 rc{pose.x+c*shift,pose.y+s*shift};
    const double a=(robot.front+robot.back)/2,b=robot.half_width,dx=center.x-rc.x,dy=center.y-rc.y;
    const bool separated=std::abs(dx)>a*std::abs(c)+b*std::abs(s)+hx||
        std::abs(dy)>a*std::abs(s)+b*std::abs(c)+hy||
        std::abs(c*dx+s*dy)>a+hx*std::abs(c)+hy*std::abs(s)||
        std::abs(-s*dx+c*dy)>b+hx*std::abs(s)+hy*std::abs(c);
    if(!separated)return 0;
    const auto r=robot_corners(pose,robot);
    const std::array<Point2,4> q{{{box.xmin,box.ymin},{box.xmax,box.ymin},{box.xmax,box.ymax},{box.xmin,box.ymax}}};
    double result=100;
    for(std::size_t i=0;i<4;++i)for(std::size_t j=0;j<4;++j){
        result=std::min(result,point_segment(r[i],q[j],q[(j+1)%4]));
        result=std::min(result,point_segment(q[i],r[j],r[(j+1)%4]));
    }
    return result;
}
std::string frame_name(int frame){std::ostringstream out;out<<std::setw(4)<<std::setfill('0')<<frame;return out.str()+".svg";}
void svg(const fs::path& path,Pose2 pose,RobotFootprint robot,Point2 target,const std::vector<Box>& boxes,
         const std::vector<Point2>& trail,const NavigationResult& result,double time){
    std::ofstream file(path);file<<std::setprecision(8);
    file<<"<svg xmlns='http://www.w3.org/2000/svg' viewBox='-0.8 -2 4.8 4'><rect x='-0.8' y='-2' width='4.8' height='4' fill='#101820'/>";
    file<<"<g transform='scale(1,-1)' stroke-width='.015'>";
    for(auto box:boxes)file<<"<rect x='"<<box.xmin<<"' y='"<<box.ymin<<"' width='"<<box.xmax-box.xmin<<"' height='"<<box.ymax-box.ymin<<"' fill='#f16e65'/>";
    file<<"<polyline fill='none' stroke='#6e91b6' points='";for(auto p:trail)file<<p.x<<','<<p.y<<' ';file<<"'/>";
    file<<"<polyline fill='none' stroke='#27dce0' points='";
    for(auto p:result.plan.trajectory.path){const auto world=transform_point({p.x,p.y},pose);file<<world.x<<','<<world.y<<' ';}file<<"'/>";
    file<<"<polygon fill='"<<(result.moving_allowed?"#58d993":"#e4bc58")<<"' stroke='#e7f4fc' points='";
    for(auto p:robot_corners(pose,robot))file<<p.x<<','<<p.y<<' ';file<<"'/>";
    const auto nose=transform_point({robot.front+.12,0},pose);
    file<<"<line x1='"<<pose.x<<"' y1='"<<pose.y<<"' x2='"<<nose.x<<"' y2='"<<nose.y<<"' stroke='white'/>";
    file<<"<circle cx='"<<target.x<<"' cy='"<<target.y<<"' r='.08' fill='#cc8bff'/></g>";
    file<<"<text x='-.65' y='-1.75' fill='white' font-size='.1' font-family='monospace'>t="<<time<<" s | "<<result.reason<<"</text></svg>";
    if(!file)throw std::runtime_error("cannot write navigation SVG");
}
void run(const Scenario& scene,const fs::path& root){
    const auto folder=root/scene.name;fs::create_directory(folder);
    RobotFootprint robot{.25,.25,.2,.04};MotionLimits limits;limits.max_linear=.3;limits.max_lateral=.25;limits.max_yaw=.8;
    NavigationConfig config;config.avoidance.allow_detour=scene.detour;
    config.avoidance.trajectory.obstacle_speed_bound_mps=scene.name=="crossing_person"?.5:0.;
    FollowNavigator navigator(robot,limits,config);LaserConfig laser_config;
    Pose2 pose;Velocity2 velocity;Point2 target{2.8,0};std::vector<Point2> trail;
    std::ofstream csv(folder/"results.csv"),html(folder/"index.html");
    std::ofstream parameters(folder/"parameters.txt");
    parameters<<"SYNTHETIC CONFIGURATION; not R550 hardware calibration\nfootprint_front_back_halfwidth_margin_m "<<robot.front<<' '<<robot.back<<' '<<robot.half_width<<' '<<robot.margin
        <<"\nmax_linear_lateral_yaw "<<limits.max_linear<<' '<<limits.max_lateral<<' '<<limits.max_yaw
        <<"\naccel_linear_yaw "<<limits.accel_linear<<' '<<limits.accel_yaw<<"\nbrake_linear_yaw "<<limits.brake_linear<<' '<<limits.brake_yaw
        <<"\nreaction_s "<<config.avoidance.trajectory.reaction_s<<"\nobstacle_speed_bound_mps "<<config.avoidance.trajectory.obstacle_speed_bound_mps
        <<"\ndetour_enabled "<<scene.detour<<"\ncontrol_dt_s .1\n"
        <<"follow_distance_deadband_gains "<<config.follow.desired_distance_m<<' '<<config.follow.distance_deadband_m<<' '
        <<config.follow.bearing_deadband_rad<<' '<<config.follow.distance_gain<<' '<<config.follow.bearing_gain<<' '<<config.follow.turn_first_angle_rad
        <<"\ntrajectory_horizon_step_tracking_margin "<<config.avoidance.trajectory.horizon_s<<' '<<config.avoidance.trajectory.time_step_s<<' '<<config.avoidance.trajectory.tracking_margin_m
        <<"\ngrid_xmin_ymin_xmax_ymax_resolution "<<laser_config.grid.xmin<<' '<<laser_config.grid.ymin<<' '<<laser_config.grid.xmax<<' '<<laser_config.grid.ymax<<' '<<laser_config.grid.resolution
        <<"\nlaser_age_ray_fraction_minrays_infclear_undeskewduration "<<laser_config.max_age_s<<' '<<laser_config.max_ray_m<<' '<<laser_config.min_usable_fraction<<' '
        <<laser_config.min_usable_rays<<' '<<laser_config.positive_infinity_is_clear<<' '<<laser_config.max_undeskewed_duration_s
        <<"\nconfirmation_count_duration "<<config.clear_scan_count<<' '<<config.clear_duration_s
        <<"\ninput_maxages_target_scan_velocity_reference_skew "<<config.follow.max_target_age_s<<' '<<config.max_scan_age_s<<' '<<config.max_velocity_age_s<<' '<<config.max_grid_reference_skew_s
        <<"\nplanner_samples_detour_clearance_lookahead_clearance_weight "<<config.avoidance.linear_samples<<' '<<config.avoidance.lateral_samples<<' '
        <<config.avoidance.yaw_samples<<' '<<config.avoidance.detour_clearance_m<<' '<<config.avoidance.lookahead_m<<' '<<config.avoidance.clearance_weight<<"\n";
    csv<<"step,time_s,x_m,y_m,yaw_rad,cmd_x_mps,cmd_y_mps,cmd_yaw_rps,measured_speed_mps,target_range_m,clearance_m,allowed,reason\n";
    csv<<std::setprecision(12);
    html<<"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>"<<scene.name<<"</title><style>body{background:#101820;color:#edf5ff;font:16px system-ui;max-width:1000px;margin:30px auto;padding:15px}img{width:100%}input{width:70%}button{padding:8px}</style><h1>"<<scene.name<<"</h1><p>合成闭环验证；紫色目标，红色障碍，青色候选轨迹。车体尺寸与制动参数为合成配置。</p><button id='play'>播放/暂停</button><input id='slider' type='range' min='0' max='"<<scene.steps-1<<"' value='0'><img id='view'><script>const names=[";
    double min_clearance=100,max_lateral=0;std::size_t moving=0,stops=0;bool crossing_stopped=false,resumed=false;
    for(int frame=0;frame<scene.steps;++frame){
        const double time=frame*.1;
        auto boxes=obstacles(scene,time);
        // The target itself is also a physical obstacle, never masked from lidar.
        auto all_boxes=boxes;all_boxes.push_back({target.x-.15,target.y-.15,target.x+.15,target.y+.15});
        auto ranges=scan_ranges(pose,all_boxes);
        if(scene.name=="sensor_dropout"&&time>=2&&time<3)std::fill(ranges.begin(),ranges.end(),std::numeric_limits<float>::quiet_NaN());
        LaserScanView raw{ranges.data(),ranges.size(),-pi,2*pi/720,.02,5,time,0};
        auto scan_name=fs::path(frame_name(frame));scan_name.replace_extension(".scan");
        write_laser_scan(folder/scan_name,raw);
        const auto laser=laser_to_grid(raw,{},time,laser_config);
        const auto relative=transform_point(target,inverse(pose));
        FollowTarget follow{relative,time,true};
        if(scene.name=="target_loss"&&time>=2&&time<3)follow.confirmed=false;
        const auto result=navigator.step(follow,laser,{velocity,time},time);
        if(result.moving_allowed&&!stopped(result.command))++moving;else ++stops;
        if(scene.name=="crossing_person"&&time>2&&time<5&&!result.moving_allowed)crossing_stopped=true;
        if(scene.name=="crossing_person"&&time>6.5&&result.moving_allowed)resumed=true;
        double clearance=100;for(auto box:all_boxes)clearance=std::min(clearance,physical_clearance(pose,robot,box));
        min_clearance=std::min(min_clearance,clearance);max_lateral=std::max(max_lateral,std::abs(pose.y));
        csv<<frame<<','<<time<<','<<pose.x<<','<<pose.y<<','<<pose.yaw<<','<<result.command.x<<','<<result.command.y<<','<<result.command.yaw<<','
           <<speed(velocity)<<','<<std::hypot(relative.x,relative.y)<<','<<clearance<<','<<result.moving_allowed<<','<<result.reason<<'\n';
        trail.push_back({pose.x,pose.y});svg(folder/frame_name(frame),pose,robot,target,boxes,trail,result,time);
        if(frame)html<<',';html<<'\''<<frame_name(frame)<<'\'';
        // Discrete actuator model: one control interval of response delay; then
        // acceleration/braking limits. Check continuous world boxes at 20 ms.
        for(int substep=1;substep<=5;++substep){
            pose=integrate(pose,velocity,.02);
            auto future=obstacles(scene,time+substep*.02);future.push_back(all_boxes.back());
            for(auto box:future)min_clearance=std::min(min_clearance,physical_clearance(pose,robot,box));
        }
        velocity=stopped(result.command)?brake_velocity(velocity,limits,.1):approach_velocity(velocity,result.command,limits,.1);
    }
    html<<"];const s=document.getElementById('slider'),v=document.getElementById('view');function show(){v.src=names[Number(s.value)];}s.oninput=show;let t=null;document.getElementById('play').onclick=()=>{if(t){clearInterval(t);t=null;}else{t=setInterval(()=>{s.value=(Number(s.value)+1)%names.length;show();},100);}};show();</script></html>";
    const double target_range=distance({pose.x,pose.y},target);
    if(!csv||!html||!parameters)throw std::runtime_error("navigation output failed");
    if(min_clearance<=1e-6)throw std::runtime_error(scene.name+": synthetic physical collision");
    if(scene.name=="static_detour"&&(pose.x<1.3||max_lateral<.3))throw std::runtime_error("static detour did not pass the obstacle");
    if(scene.name=="crossing_person"&&(!crossing_stopped||!resumed))throw std::runtime_error("crossing stop/resume contract failed");
    if(scene.name=="clear_follow"&&target_range>1.4)throw std::runtime_error("distance feedback did not converge");
    std::ofstream summary(folder/"summary.txt");summary<<std::setprecision(8)<<"scenario="<<scene.name<<"\nframes="<<scene.steps<<"\nminimum_physical_clearance_m="<<min_clearance<<"\nmaximum_lateral_offset_m="<<max_lateral<<"\nfinal_target_range_m="<<target_range<<"\nmoving_frames="<<moving<<"\nstopped_frames="<<stops<<"\n";
    summary.flush();if(!summary)throw std::runtime_error("cannot write navigation summary");
    std::cout<<scene.name<<": min clearance="<<min_clearance<<" m, final target range="<<target_range<<" m, lateral="<<max_lateral<<" m\n";
}
}
int main(int argc,char** argv){
    try{
        if(argc!=2){std::cerr<<"Usage: navigation_replay NEW_OUTPUT_DIR\n";return 2;}
        const fs::path root=argv[1];if(fs::exists(root)&&(!fs::is_directory(root)||!fs::is_empty(root)))throw std::runtime_error("output directory must be new or empty");
        fs::create_directories(root);
        std::ofstream index(root/"index.html");index<<"<!doctype html><meta charset='utf-8'><title>激光避障与跟随验证</title><h1>合成闭环场景</h1><p>参数为合成配置，未连接真实机器人。</p><ul>";
        for(const auto& scene:std::vector<Scenario>{{"clear_follow",false,100},{"crossing_person",false,110},{"static_detour",true,150},{"blocked_passage",true,70},{"sensor_dropout",false,100},{"target_loss",false,100}}){
            run(scene,root);index<<"<li><a href='"<<scene.name<<"/index.html'>"<<scene.name<<"</a></li>";
        }
        index<<"</ul>";if(!index)throw std::runtime_error("cannot write navigation index");
        std::cout<<"PASS: 6 synthetic closed-loop scenarios. Viewer: "<<(root/"index.html")<<'\n';
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
