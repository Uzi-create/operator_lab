#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// ABI: 0 success, 1 invalid argument, 2 allocation/internal error,
// 3 statistics capacity too small (outputs may be partially written; retry).
// Depth values may contain NaN/Inf; they are holes. Region rows contain
// count,min_x,min_y,max_x,max_y,min_depth,max_depth,mean_depth.
extern "C" int depth_connected_components(const double* depth, size_t width, size_t height,
    double scale, double min_depth, double max_depth, double absolute_jump,
    double relative_jump, int connectivity, size_t min_area, int32_t* labels,
    double* stats, size_t stats_capacity, int32_t* region_count) noexcept {
    if(!depth || !labels || !region_count || !width || !height || width>16384 || height>16384 ||
       width>16000000/height || !min_area || min_area>width*height ||
       !std::isfinite(scale) || scale<=0 || !std::isfinite(min_depth) || min_depth<0 ||
       !std::isfinite(max_depth) || max_depth<=min_depth || max_depth>1e100 ||
       !std::isfinite(absolute_jump) || absolute_jump<0 || absolute_jump>1e100 ||
       !std::isfinite(relative_jump) || relative_jump<0 || relative_jump>1000 ||
       (connectivity!=4 && connectivity!=8) || stats_capacity>width*height ||
       (stats_capacity && !stats)) return 1;
    try {
        const size_t n=width*height;
        std::vector<double> z(n);
        std::vector<uint8_t> seen(n,0);
        std::vector<size_t> queue(n);
        for(size_t i=0;i<n;++i) {
            const double value=depth[i]*scale;
            z[i]=(std::isfinite(value) && value>min_depth && value<=max_depth) ? value : 0.;
        }
        std::fill_n(labels,n,int32_t(0));
        int32_t regions=0;
        constexpr int dx[8]={-1,1,0,0,-1,1,-1,1};
        constexpr int dy[8]={0,0,-1,1,-1,-1,1,1};
        for(size_t root=0;root<n;++root) {
            if(seen[root] || z[root]==0.) continue;
            size_t head=0,tail=1;
            queue[0]=root;seen[root]=1;
            size_t min_x=root%width,max_x=min_x,min_y=root/width,max_y=min_y;
            double low=z[root],high=low,sum=0.;
            while(head<tail) {
                const size_t p=queue[head++],x=p%width,y=p/width;
                const double a=z[p];
                min_x=std::min(min_x,x);max_x=std::max(max_x,x);
                min_y=std::min(min_y,y);max_y=std::max(max_y,y);
                low=std::min(low,a);high=std::max(high,a);sum+=a;
                for(int k=0;k<connectivity;++k) {
                    const auto nx=static_cast<int64_t>(x)+dx[k];
                    const auto ny=static_cast<int64_t>(y)+dy[k];
                    if(nx<0 || ny<0 || static_cast<size_t>(nx)>=width || static_cast<size_t>(ny)>=height) continue;
                    const size_t q=static_cast<size_t>(ny)*width+static_cast<size_t>(nx);
                    const double b=z[q];
                    if(seen[q] || b==0. || std::abs(a-b)>absolute_jump+relative_jump*std::min(a,b)) continue;
                    seen[q]=1;queue[tail++]=q;
                }
            }
            if(tail<min_area) continue;
            if(static_cast<size_t>(regions)>=stats_capacity) return 3;
            ++regions;
            for(size_t i=0;i<tail;++i) labels[queue[i]]=regions;
            double* row=stats+8*static_cast<size_t>(regions-1);
            row[0]=static_cast<double>(tail);row[1]=static_cast<double>(min_x);
            row[2]=static_cast<double>(min_y);row[3]=static_cast<double>(max_x);
            row[4]=static_cast<double>(max_y);row[5]=low;row[6]=high;
            row[7]=sum/static_cast<double>(tail);
        }
        *region_count=regions;
        return 0;
    } catch(...) { return 2; }
}
