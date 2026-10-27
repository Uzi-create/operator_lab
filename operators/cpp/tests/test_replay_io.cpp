#include <operator_lab/replay_io.hpp>
#include <chrono>
#include <iostream>
#include <iterator>

using namespace operator_lab::rgbd;
namespace fs=std::filesystem;
namespace {
std::size_t checks=0;
#define CHECK(expr) do {++checks;if(!(expr))throw std::runtime_error(std::string(__FILE__)+":"+std::to_string(__LINE__)+" " #expr);}while(false)
template<class F> void throws(F function) {bool caught=false;try{function();}catch(const std::exception&){caught=true;}CHECK(caught);}
struct Temporary {
    fs::path path=fs::temp_directory_path()/("operator-rgbd-io-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary(){if(!fs::create_directory(path))throw std::runtime_error("cannot create unique test directory");}
    ~Temporary(){std::error_code error;fs::remove_all(path,error);}
};
void text(const fs::path& path,const std::string& value) {std::ofstream file(path,std::ios::binary);file<<value;}
void io_tests() {
    Temporary temp;
    // Golden bytes are independent of our writer; whitespace and # in the
    // first pixel are raster data, never header trivia.
    std::string ppm="P6\r\n# comment\r\n1 1\r\n255\r\n";
    ppm.append({char(10),char(32),char(35)});text(temp.path/"gold.ppm",ppm);
    const auto image=read_rgb_ppm(temp.path/"gold.ppm");
    CHECK(image.width==1&&image.height==1&&image.pixels==std::vector<std::uint8_t>({10,32,35}));
    write_rgb_ppm(temp.path/"roundtrip.ppm",image);
    CHECK(read_rgb_ppm(temp.path/"roundtrip.ppm").pixels==image.pixels);
    std::string pgm="P5\n4 1\n65535\n";
    for(unsigned byte:{0,0,0,1,1,0,255,255})pgm.push_back(static_cast<char>(byte));
    text(temp.path/"gold.pgm",pgm);
    const auto depth=read_depth_pgm16(temp.path/"gold.pgm");
    CHECK(depth.pixels==std::vector<std::uint16_t>({0,1,256,65535}));
    write_depth_pgm16(temp.path/"roundtrip.pgm",depth);
    std::ifstream roundtrip(temp.path/"roundtrip.pgm",std::ios::binary);
    CHECK(std::string(std::istreambuf_iterator<char>(roundtrip),{})==pgm);
    text(temp.path/"bad.ppm","P6\n2 2\n255\nabc");throws([&]{read_rgb_ppm(temp.path/"bad.ppm");});
    text(temp.path/"bad.ppm","P6\n-1 2\n255\nabc");throws([&]{read_rgb_ppm(temp.path/"bad.ppm");});
    text(temp.path/"bad.pgm","P5\n1 1\n255\nxx");throws([&]{read_depth_pgm16(temp.path/"bad.pgm");});
    text(temp.path/"bad.pgm","P5\n9999999999 2\n65535\n");throws([&]{read_depth_pgm16(temp.path/"bad.pgm");});
    text(temp.path/"calibration.txt","4 1 100 101 2 0 0.001\n");
    const auto calibration=read_replay_calibration(temp.path/"calibration.txt");
    CHECK(calibration.camera.width==4&&calibration.camera.fy==101&&calibration.depth_scale_to_m==.001);
    text(temp.path/"calibration.txt","4 1 100 101 2 0 0\n");throws([&]{read_replay_calibration(temp.path/"calibration.txt");});
    text(temp.path/"frames.txt","# header\n0 0 0 0 \"file with spaces.ppm\" image.pgm 1 0 0 0 1 0 0 0 1 .2 0 0 2 0\n");
    const auto frames=read_replay_manifest(temp.path/"frames.txt");
    CHECK(frames.size()==1&&frames[0].rgb_path.filename()=="file with spaces.ppm");
    CHECK(frames[0].selection->u==2&&frames[0].fixed_from_camera.translation_m.x==.2);
    text(temp.path/"frames.txt","0 0 0 0 a b 1 0 0 0 1 0 0 0 1 0 0 0 -1 2\n");
    throws([&]{read_replay_manifest(temp.path/"frames.txt");});
    text(temp.path/"frames.txt","0 0 0 0 a b -1 0 0 0 1 0 0 0 1 0 0 0 -1 -1\n");
    throws([&]{read_replay_manifest(temp.path/"frames.txt");});
    text(temp.path/"frames.txt","0 0 0 0 a b 1 0 0 0 1 0 0 0 1 0 0 0 -1 -1 extra\n");
    throws([&]{read_replay_manifest(temp.path/"frames.txt");});
    text(temp.path/"frames.txt","# no frames\n");throws([&]{read_replay_manifest(temp.path/"frames.txt");});
    write_rgb_bmp(temp.path/"pixel.bmp",image);
    std::ifstream bmp(temp.path/"pixel.bmp",std::ios::binary);
    const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(bmp),{}};
    CHECK(bytes.size()==58&&bytes[0]=='B'&&bytes[1]=='M');
    CHECK(bytes[54]==35&&bytes[55]==32&&bytes[56]==10&&bytes[57]==0); // BGR plus row padding.
}
}
int main(){try{io_tests();std::cout<<"PASS: "<<checks<<" C++ replay I/O checks\n";}
catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
