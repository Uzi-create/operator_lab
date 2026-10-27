#include <operator_lab/replay_io.hpp>
#include <chrono>
#include <iostream>

using namespace operator_lab::rgbd;
namespace fs=std::filesystem;
namespace {
void new_directory(const fs::path& path) {
    if(fs::exists(path)&&(!fs::is_directory(path)||!fs::is_empty(path)))
        throw std::runtime_error("choose a new or empty output directory: "+path.string());
    fs::create_directories(path);
}
std::string number(std::size_t frame) {
    std::ostringstream stream;stream<<std::setw(6)<<std::setfill('0')<<frame;return stream.str();
}
void generate_demo(const fs::path& directory) {
    new_directory(directory);
    std::ofstream calibration(directory/"calibration.txt");calibration<<"80 48 70 70 40 24 0.001\n";
    std::ofstream manifest(directory/"frames.txt");manifest<<std::setprecision(17);
    manifest<<"# rgb_s depth_s pose_s now_s RGB_PPM DEPTH_PGM R_row_major[9] t_m[3] seed_u seed_v\n";
    for(std::size_t frame=0;frame<34;++frame) {
        RgbImage rgb{80,48,std::vector<std::uint8_t>(80*48*3,16)};
        DepthImage depth{80,48,std::vector<std::uint16_t>(80*48,0)};
        auto box=[&](std::size_t x,std::uint16_t z,std::array<std::uint8_t,3> color) {
            for(std::size_t y=15;y<33;++y) for(std::size_t u=x;u<x+8;++u) {
                depth.pixels[y*80+u]=z;
                for(std::size_t c=0;c<3;++c) rgb.pixels[(y*80+u)*3+c]=color[c];
            }
        };
        const auto target_x=20+frame/3;
        const bool hidden=(frame>=6&&frame<=8)||(frame>=18&&frame<=28);
        if(!hidden) box(target_x,2000,{230,35,35});
        const auto other_x=static_cast<std::size_t>(std::max(0,60-static_cast<int>(frame)*2));
        box(frame>=6&&frame<=8?target_x:other_x,frame>=6&&frame<=8?1200:2000,{35,35,230});
        const auto name=number(frame);
        write_rgb_ppm(directory/(name+".ppm"),rgb);write_depth_pgm16(directory/(name+".pgm"),depth);
        const double time=static_cast<double>(frame)*.1;
        manifest<<time<<' '<<time<<' '<<time<<' '<<time<<' '<<std::quoted(name+".ppm")<<' '
                <<std::quoted(name+".pgm")<<" 1 0 0 0 1 0 0 0 1 "<<frame*.01<<" 0 0 ";
        if(frame==0||frame==30) manifest<<target_x+2<<" 23\n";else manifest<<"-1 -1\n";
    }
    if(!calibration||!manifest) throw std::runtime_error("cannot finish demo metadata");
    std::cout<<"Generated 34 SYNTHETIC registered RGB-D frames in "<<directory<<'\n';
}
void paint(RgbImage& image,int x,int y,std::array<std::uint8_t,3> color) {
    if(x<0||y<0||x>=static_cast<int>(image.width)||y>=static_cast<int>(image.height)) return;
    const auto offset=(static_cast<std::size_t>(y)*image.width+static_cast<std::size_t>(x))*3;
    for(std::size_t c=0;c<3;++c) image.pixels[offset+c]=color[c];
}
RgbImage preview(const RgbImage& rgb,const DepthImage& depth,const ReplayCalibration& calibration,
                 const ReplayFrame& frame,const TrackerResult& result) {
    RgbImage out{rgb.width*2,rgb.height,std::vector<std::uint8_t>(rgb.width*2*rgb.height*3,16)};
    for(std::size_t y=0;y<rgb.height;++y) for(std::size_t x=0;x<rgb.width;++x) {
        const auto source=(y*rgb.width+x)*3,destination=(y*out.width+x)*3;
        for(std::size_t c=0;c<3;++c) out.pixels[destination+c]=rgb.pixels[source+c];
        const double z=depth.pixels[y*rgb.width+x]*calibration.depth_scale_to_m;
        if(z>0) {
            const double value=std::clamp(z/8.,0.,1.);
            paint(out,static_cast<int>(x+rgb.width),static_cast<int>(y),
                  {static_cast<std::uint8_t>(255*(1-value)),static_cast<std::uint8_t>(180*value),
                   static_cast<std::uint8_t>(255*value)});
        }
    }
    const std::array<std::uint8_t,3> color=result.confirmed?std::array<std::uint8_t,3>{20,255,80}:
                                                                        std::array<std::uint8_t,3>{255,180,20};
    if(result.bounds) {
        const auto box=*result.bounds;
        for(std::size_t x=box.x;x<box.x+box.width;++x) {
            paint(out,static_cast<int>(x),static_cast<int>(box.y),color);
            paint(out,static_cast<int>(x),static_cast<int>(box.y+box.height-1),color);
        }
        for(std::size_t y=box.y;y<box.y+box.height;++y) {
            paint(out,static_cast<int>(box.x),static_cast<int>(y),color);
            paint(out,static_cast<int>(box.x+box.width-1),static_cast<int>(y),color);
        }
    }
    if(result.prediction) {
        // Invert T_fixed_from_camera only for visualizing the prediction.
        const auto& r=frame.fixed_from_camera.rotation;
        const auto p=result.prediction->position_m,t=frame.fixed_from_camera.translation_m;
        const Point3 d{p.x-t.x,p.y-t.y,p.z-t.z};
        const Point3 camera{r[0]*d.x+r[3]*d.y+r[6]*d.z,r[1]*d.x+r[4]*d.y+r[7]*d.z,r[2]*d.x+r[5]*d.y+r[8]*d.z};
        if(camera.z>0) {
            const double u=calibration.camera.fx*camera.x/camera.z+calibration.camera.cx;
            const double v=calibration.camera.fy*camera.y/camera.z+calibration.camera.cy;
            if(u>=0&&v>=0&&u<rgb.width&&v<rgb.height) for(int delta=-2;delta<=2;++delta) {
                paint(out,static_cast<int>(u)+delta,static_cast<int>(v),{0,255,255});
                paint(out,static_cast<int>(u),static_cast<int>(v)+delta,{0,255,255});
            }
        }
    }
    return out;
}
void replay(const fs::path& calibration_path,const fs::path& manifest_path,const fs::path& directory) {
    const auto calibration=read_replay_calibration(calibration_path);
    const auto frames=read_replay_manifest(manifest_path);
    new_directory(directory);
    std::ofstream csv(directory/"results.csv"),html(directory/"index.html");
    if(!csv||!html) throw std::runtime_error("cannot open replay outputs");
    csv<<"frame,stamp_s,state,reason,confirmed,fresh,regions,x_m,y_m,z_m,predicted_x_m,predicted_y_m,predicted_z_m,processing_ms\n";
    html<<R"HTML(<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RGB-D 回放诊断</title>
<style>body{background:#101820;color:#eaf1fa;font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 20px}img{width:100%;image-rendering:pixelated;border:1px solid #345}input{width:80%}pre{white-space:pre-wrap;color:#b8d5ee}button{padding:8px 18px}small{color:#a8b9ca}</style>
<h1>RGB-D 跟踪回放</h1><p>左：RGB 与目标框；右：深度。绿框：确认，橙框：确认中，青色十字：预测位置。</p>
<small>这是算法诊断输出；目标确认不代表允许机器人运动。合成示例不代表真实相机效果。</small>
<p><button id="play">播放 / 暂停</button> <input id="slider" type="range" min="0" value="0"></p><img id="view"><pre id="info"></pre>
<script>const frames=[
)HTML";
    RgbdTracker tracker;std::size_t confirmed_count=0;
    csv<<std::setprecision(12);
    for(std::size_t i=0;i<frames.size();++i) {
        const auto rgb=read_rgb_ppm(frames[i].rgb_path);const auto depth=read_depth_pgm16(frames[i].depth_path);
        if(rgb.width!=calibration.camera.width||rgb.height!=calibration.camera.height||
           depth.width!=rgb.width||depth.height!=rgb.height) throw std::runtime_error("replay image dimensions differ from calibration");
        const auto begin=std::chrono::steady_clock::now();
        const auto result=tracker.process(RgbView(rgb.pixels.data(),rgb.pixels.size(),rgb.width,rgb.height),
            DepthView<std::uint16_t>(depth.pixels.data(),depth.pixels.size(),depth.width,depth.height,calibration.depth_scale_to_m),
            calibration.camera,frames[i].fixed_from_camera,frames[i].time,frames[i].selection);
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        confirmed_count+=result.confirmed?1:0;
        csv<<i<<','<<frames[i].time.depth_stamp_s<<','<<state_name(result.state)<<','<<result.reason<<','
           <<result.confirmed<<','<<result.fresh_observation<<','<<result.region_count;
        auto point=[&](std::optional<Point3> p){if(p) csv<<','<<p->x<<','<<p->y<<','<<p->z;else csv<<",,,";};
        point(result.observed_position_m);point(result.prediction?std::optional<Point3>(result.prediction->position_m):std::nullopt);
        csv<<','<<ms<<'\n';
        const auto name=number(i)+".bmp";
        write_rgb_bmp(directory/name,preview(rgb,depth,calibration,frames[i],result));
        if(i) html<<",\n";
        html<<"{file:'"<<name<<"',state:'"<<state_name(result.state)<<"',reason:'"<<result.reason
            <<"',confirmed:"<<(result.confirmed?"true":"false")<<",regions:"<<result.region_count
            <<",stamp:"<<frames[i].time.depth_stamp_s<<"}";
    }
    html<<R"HTML(];const slider=document.getElementById('slider'),view=document.getElementById('view'),info=document.getElementById('info');slider.max=frames.length-1;
function show(){const i=Number(slider.value),f=frames[i];view.src=f.file;info.textContent=`帧 ${i+1}/${frames.length} | 时间 ${f.stamp.toFixed(3)} s\n状态 ${f.state} | 确认 ${f.confirmed} | 深度区域 ${f.regions}\n原因 ${f.reason}`;}slider.oninput=show;let timer=null;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;}else{timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%frames.length;show();},150);}};show();</script></html>)HTML";
    if(!csv||!html) throw std::runtime_error("failed writing replay outputs");
    std::cout<<"Replayed "<<frames.size()<<" frames; "<<confirmed_count<<" confirmed observations\n"
             <<"Diagnostics: "<<(directory/"index.html")<<"\nCSV: "<<(directory/"results.csv")<<'\n';
}
}
int main(int argc,char** argv) {
    try {
        if(argc==3&&std::string(argv[1])=="--generate-demo") {generate_demo(argv[2]);return 0;}
        if(argc==4) {replay(argv[1],argv[2],argv[3]);return 0;}
        std::cerr<<"Usage:\n  rgbd_replay --generate-demo NEW_INPUT_DIR\n"
                 <<"  rgbd_replay calibration.txt frames.txt NEW_OUTPUT_DIR\n";
        return argc==2&&std::string(argv[1])=="--help"?0:2;
    } catch(const std::exception& e) {std::cerr<<"Replay failed: "<<e.what()<<'\n';return 1;}
}
