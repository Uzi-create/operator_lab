#pragma once
#include "rgbd_tracker.hpp"
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace operator_lab::rgbd {
struct RgbImage { std::size_t width=0,height=0;std::vector<std::uint8_t> pixels; };
struct DepthImage { std::size_t width=0,height=0;std::vector<std::uint16_t> pixels; };
struct ReplayCalibration { Intrinsics camera;double depth_scale_to_m=0; };
struct ReplayFrame {
    FrameTimes time;
    std::filesystem::path rgb_path,depth_path;
    RigidTransform fixed_from_camera;
    std::optional<Pixel> selection;
};
namespace replay_detail {
inline std::string token(std::istream& stream) {
    std::string result;char c=0;
    while(stream.get(c)) {
        if(std::isspace(static_cast<unsigned char>(c))) continue;
        if(c=='#') {stream.ignore(std::numeric_limits<std::streamsize>::max(),'\n');continue;}
        result.push_back(c);break;
    }
    while(stream.get(c)) {
        if(std::isspace(static_cast<unsigned char>(c))) {
            if(c=='\r'&&stream.peek()=='\n') stream.get();
            break;
        }
        result.push_back(c);
    }
    if(result.empty()||!stream) throw std::runtime_error("incomplete PNM header");
    return result;
}
inline std::size_t integer(const std::string& text) {
    if(text.empty()||text.size()>10||text.find_first_not_of("0123456789")!=std::string::npos)
        throw std::runtime_error("invalid PNM integer");
    const auto value=std::stoull(text);
    if(value>std::numeric_limits<std::size_t>::max()) throw std::runtime_error("PNM integer overflow");
    return static_cast<std::size_t>(value);
}
inline void dimensions(std::size_t width,std::size_t height) {
    if(!width||!height||width>16384||height>16384||width>16000000/height)
        throw std::runtime_error("unsupported PNM dimensions");
}
inline void exact_read(std::istream& stream,char* out,std::size_t bytes) {
    stream.read(out,static_cast<std::streamsize>(bytes));
    if(static_cast<std::size_t>(stream.gcount())!=bytes) throw std::runtime_error("truncated PNM raster");
}
inline void end_of_line(std::istream& line) {
    line>>std::ws;
    if(line.peek()!=std::char_traits<char>::eof()&&line.peek()!='#')
        throw std::runtime_error("extra manifest/calibration fields");
}
}
inline RgbImage read_rgb_ppm(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    if(!file) throw std::runtime_error("cannot open RGB PPM: "+path.string());
    if(replay_detail::token(file)!="P6") throw std::runtime_error("RGB must be binary P6 PPM");
    RgbImage out;out.width=replay_detail::integer(replay_detail::token(file));
    out.height=replay_detail::integer(replay_detail::token(file));
    replay_detail::dimensions(out.width,out.height);
    if(replay_detail::token(file)!="255") throw std::runtime_error("PPM maximum must be 255");
    out.pixels.resize(out.width*out.height*3);
    replay_detail::exact_read(file,reinterpret_cast<char*>(out.pixels.data()),out.pixels.size());
    return out;
}
inline DepthImage read_depth_pgm16(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    if(!file) throw std::runtime_error("cannot open depth PGM: "+path.string());
    if(replay_detail::token(file)!="P5") throw std::runtime_error("depth must be binary P5 PGM");
    DepthImage out;out.width=replay_detail::integer(replay_detail::token(file));
    out.height=replay_detail::integer(replay_detail::token(file));
    replay_detail::dimensions(out.width,out.height);
    if(replay_detail::token(file)!="65535") throw std::runtime_error("depth PGM maximum must be 65535");
    std::vector<std::uint8_t> bytes(out.width*out.height*2);
    replay_detail::exact_read(file,reinterpret_cast<char*>(bytes.data()),bytes.size());
    out.pixels.resize(out.width*out.height);
    for(std::size_t i=0;i<out.pixels.size();++i)
        out.pixels[i]=static_cast<std::uint16_t>(static_cast<unsigned>(bytes[i*2])*256+bytes[i*2+1]);
    return out;
}
inline ReplayCalibration read_replay_calibration(const std::filesystem::path& path) {
    std::ifstream file(path);ReplayCalibration out;
    if(!(file>>out.camera.width>>out.camera.height>>out.camera.fx>>out.camera.fy>>out.camera.cx>>out.camera.cy>>out.depth_scale_to_m))
        throw std::runtime_error("calibration needs width height fx fy cx cy depth_scale_to_m");
    replay_detail::end_of_line(file);validate(out.camera);
    if(!std::isfinite(out.depth_scale_to_m)||out.depth_scale_to_m<=0) throw std::runtime_error("invalid depth scale");
    return out;
}
inline std::vector<ReplayFrame> read_replay_manifest(const std::filesystem::path& path) {
    std::ifstream file(path);
    if(!file) throw std::runtime_error("cannot open replay manifest");
    std::vector<ReplayFrame> frames;std::string line;std::size_t line_number=0;
    while(std::getline(file,line)) {
        ++line_number;std::istringstream row(line);row>>std::ws;
        if(row.peek()=='#'||row.peek()==std::char_traits<char>::eof()) continue;
        ReplayFrame frame;std::string rgb,depth;long long seed_u=-1,seed_v=-1;
        if(!(row>>frame.time.rgb_stamp_s>>frame.time.depth_stamp_s>>frame.time.pose_stamp_s>>frame.time.now_s
              >>std::quoted(rgb)>>std::quoted(depth))) throw std::runtime_error("invalid manifest row "+std::to_string(line_number));
        for(auto& value:frame.fixed_from_camera.rotation)
            if(!(row>>value)) throw std::runtime_error("manifest row needs row-major 3x3 rotation");
        auto& t=frame.fixed_from_camera.translation_m;
        if(!(row>>t.x>>t.y>>t.z>>seed_u>>seed_v)) throw std::runtime_error("manifest row needs translation and selection");
        replay_detail::end_of_line(row);validate(frame.fixed_from_camera);
        for(double value:{frame.time.rgb_stamp_s,frame.time.depth_stamp_s,frame.time.pose_stamp_s,frame.time.now_s})
            if(!std::isfinite(value)) throw std::runtime_error("nonfinite manifest timestamp");
        if((seed_u==-1)!=(seed_v==-1)||seed_u< -1||seed_v< -1||seed_u>16383||seed_v>16383)
            throw std::runtime_error("selection must be two image coordinates or -1 -1");
        if(seed_u>=0) frame.selection=Pixel{static_cast<std::size_t>(seed_u),static_cast<std::size_t>(seed_v)};
        if(rgb.empty()||depth.empty()) throw std::runtime_error("empty frame filename");
        frame.rgb_path=path.parent_path()/rgb;frame.depth_path=path.parent_path()/depth;
        frames.push_back(std::move(frame));
    }
    if(frames.empty()) throw std::runtime_error("replay manifest is empty");
    return frames;
}
inline void write_rgb_ppm(const std::filesystem::path& path,const RgbImage& image) {
    replay_detail::dimensions(image.width,image.height);
    if(image.pixels.size()!=image.width*image.height*3) throw std::invalid_argument("RGB output size mismatch");
    std::ofstream file(path,std::ios::binary);file<<"P6\n"<<image.width<<' '<<image.height<<"\n255\n";
    file.write(reinterpret_cast<const char*>(image.pixels.data()),static_cast<std::streamsize>(image.pixels.size()));
    if(!file) throw std::runtime_error("cannot write RGB PPM");
}
inline void write_depth_pgm16(const std::filesystem::path& path,const DepthImage& image) {
    replay_detail::dimensions(image.width,image.height);
    if(image.pixels.size()!=image.width*image.height) throw std::invalid_argument("depth output size mismatch");
    std::ofstream file(path,std::ios::binary);file<<"P5\n"<<image.width<<' '<<image.height<<"\n65535\n";
    for(auto value:image.pixels) {file.put(static_cast<char>(value>>8));file.put(static_cast<char>(value&255));}
    if(!file) throw std::runtime_error("cannot write depth PGM");
}

// Browser-readable, uncompressed 24-bit BMP; independent of platform endianness.
inline void write_rgb_bmp(const std::filesystem::path& path,const RgbImage& image) {
    if(!image.width||!image.height||image.width>32768||image.height>16384||
       image.width>32000000/image.height||image.pixels.size()!=image.width*image.height*3)
        throw std::invalid_argument("invalid BMP image");
    const std::size_t stride=(image.width*3+3)/4*4, bytes=54+stride*image.height;
    std::ofstream file(path,std::ios::binary);
    auto little=[&](std::size_t value,int n){for(int i=0;i<n;++i) file.put(static_cast<char>((value>>(i*8))&255));};
    file<<"BM";little(bytes,4);little(0,4);little(54,4);little(40,4);
    little(image.width,4);little(image.height,4);little(1,2);little(24,2);little(0,4);
    little(stride*image.height,4);little(2835,4);little(2835,4);little(0,4);little(0,4);
    for(std::size_t row=image.height;row>0;--row) {
        for(std::size_t x=0;x<image.width;++x) {
            const auto offset=((row-1)*image.width+x)*3;
            file.put(static_cast<char>(image.pixels[offset+2]));file.put(static_cast<char>(image.pixels[offset+1]));
            file.put(static_cast<char>(image.pixels[offset]));
        }
        for(std::size_t pad=image.width*3;pad<stride;++pad) file.put(0);
    }
    if(!file) throw std::runtime_error("cannot write BMP");
}
} // namespace operator_lab::rgbd
