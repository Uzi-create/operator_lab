#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// Match NumPy's separate divide/multiply/add at discrete pixel boundaries.
// Fast-math is not supported: finite validation is part of the exported ABI.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif

namespace perception_detail {
inline bool overlap(const void* p,size_t a,const void* q,size_t b) {
    if(!a || !b) return false;
    const auto x=reinterpret_cast<uintptr_t>(p),y=reinterpret_cast<uintptr_t>(q);
    return x>=y ? x-y<b : y-x<a;
}
inline bool aligned(const void* p) {return reinterpret_cast<uintptr_t>(p)%8==0;}
inline bool input_ok(const double* p,size_t n) {
    if((!p && n) || !aligned(p) || n>size_t(std::numeric_limits<int64_t>::max()) ||
        n>std::numeric_limits<size_t>::max()/(3*sizeof(double))) return false;
    for(size_t i=0;i<3*n;++i) if(!std::isfinite(p[i])) return false;
    return true;
}
inline bool buffers_ok(const double* p,size_t n,void* const* outputs,size_t count,size_t size) {
    const size_t input_bytes=n*3*sizeof(double),output_bytes=size*8;
    for(size_t i=0;i<count;++i) {
        if(!outputs[i] || !aligned(outputs[i]) || overlap(p,input_bytes,outputs[i],output_bytes)) return false;
        for(size_t j=0;j<i;++j) if(overlap(outputs[i],output_bytes,outputs[j],output_bytes)) return false;
    }
    return true;
}
inline size_t cell(const double* p,double xmin,double ymin,double xmax,double ymax,
                   double resolution,size_t nx,size_t ny) {
    if(p[0]<xmin || p[0]>=xmax || p[1]<ymin || p[1]>=ymax) return nx*ny;
    const size_t x=std::min(size_t(std::floor((p[0]-xmin)/resolution)),nx-1);
    const size_t y=std::min(size_t(std::floor((p[1]-ymin)/resolution)),ny-1);
    return y*nx+x;
}
}

// Buffers must have their declared extents and remain valid for the call.
// Return 1 on invalid inputs/overlap, with no output writes; 0 on success.
extern "C" int raster_depth(const double* points,size_t n,int width,int height,
                            double fx,double fy,double cx,double cy,
                            double* depth,int64_t* source_index) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if(width<=0 || height<=0 || size_t(width)>16777216u/size_t(height) ||
        !std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) || !std::isfinite(cy) || fx<=0 || fy<=0) return 1;
    const size_t size=size_t(width)*size_t(height);
    if(!perception_detail::input_ok(points,n)) return 1;
    void* outputs[]={depth,source_index};
    if(!perception_detail::buffers_ok(points,n,outputs,2,size)) return 1;
    std::fill(depth,depth+size,0.);
    std::fill(source_index,source_index+size,int64_t(-1));
    for(size_t i=0;i<n;++i) {
        const double* p=points+3*i;
        if(p[2]<=0) continue;
        const double u=(p[0]/p[2])*fx+cx,v=(p[1]/p[2])*fy+cy;
        if(!std::isfinite(u) || !std::isfinite(v) || u<-.5 || u>=width-.5 || v<-.5 || v>=height-.5) continue;
        const size_t x=std::min(size_t(std::floor(u+.5)),size_t(width-1));
        const size_t y=std::min(size_t(std::floor(v+.5)),size_t(height-1));
        const size_t index=y*size_t(width)+x;
        // Ascending input traversal preserves the first index on equal depth.
        if(source_index[index]<0 || p[2]<depth[index]) {
            depth[index]=p[2];source_index[index]=int64_t(i);
        }
    }
    return 0;
}

extern "C" int raster_elevation(const double* points,size_t n,
    double xmin,double ymin,double xmax,double ymax,double resolution,int nx,int ny,
    int64_t* counts,double* low,double* high,double* mean) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if(nx<=0 || ny<=0 || size_t(nx)>4000000u/size_t(ny) ||
       !std::isfinite(xmin) || !std::isfinite(ymin) || !std::isfinite(xmax) || !std::isfinite(ymax) ||
       !std::isfinite(resolution) || resolution<=0 || xmin>=xmax || ymin>=ymax) return 1;
    const double gx=(xmax-xmin)/resolution,gy=(ymax-ymin)/resolution;
    if(!std::isfinite(gx) || !std::isfinite(gy) || std::ceil(gx)!=nx || std::ceil(gy)!=ny) return 1;
    if(!perception_detail::input_ok(points,n)) return 1;
    const size_t size=size_t(nx)*size_t(ny);
    void* outputs[]={counts,low,high,mean};
    if(!perception_detail::buffers_ok(points,n,outputs,4,size)) return 1;
    std::fill(counts,counts+size,int64_t(0));
    std::fill(low,low+size,std::numeric_limits<double>::infinity());
    std::fill(high,high+size,-std::numeric_limits<double>::infinity());
    std::fill(mean,mean+size,0.);
    for(size_t i=0;i<n;++i) {
        const double* p=points+3*i;
        const size_t index=perception_detail::cell(p,xmin,ymin,xmax,ymax,resolution,nx,ny);
        if(index==size) continue;
        ++counts[index];low[index]=std::min(low[index],p[2]);high[index]=std::max(high[index],p[2]);
    }
    for(size_t i=0;i<n;++i) {
        const double* p=points+3*i;
        const size_t index=perception_detail::cell(p,xmin,ymin,xmax,ymax,resolution,nx,ny);
        if(index<size) mean[index]+=p[2]/double(counts[index]);
    }
    bool overflow=false;
    for(size_t k=0;k<size;++k) overflow |= !std::isfinite(mean[k]);
    if(overflow) {
        // Rare DBL_MAX rounding case: same scaled fallback as NumPy reference.
        std::fill(mean,mean+size,0.);
        for(size_t i=0;i<n;++i) {
            const double* p=points+3*i;
            const size_t index=perception_detail::cell(p,xmin,ymin,xmax,ymax,resolution,nx,ny);
            if(index==size) continue;
            const double scale=std::max(std::abs(low[index]),std::abs(high[index]));
            mean[index]+=scale>0?p[2]/scale:0.;
        }
        for(size_t k=0;k<size;++k) if(counts[k])
            mean[k]=mean[k]/double(counts[k])*std::max(std::abs(low[k]),std::abs(high[k]));
    }
    for(size_t k=0;k<size;++k) if(!counts[k])
        low[k]=high[k]=mean[k]=std::numeric_limits<double>::quiet_NaN();
    return 0;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
