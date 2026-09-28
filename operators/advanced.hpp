// Internal C++17 kernels; run_operator validates arguments and catches exceptions.
#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace advanced {
// Double precision box means, including signed intermediate coefficient images.
inline std::vector<double> box(const std::vector<double>& src, size_t w, size_t h, size_t r) {
    const size_t s = w+1;
    if (s > std::numeric_limits<size_t>::max()/(h+1)) throw std::bad_alloc();
    std::vector<double> integral(s*(h+1), 0), out(w*h);
    for (size_t y=0; y<h; ++y) {
        double row=0;
        for (size_t x=0; x<w; ++x) {
            row += src[y*w+x];
            integral[(y+1)*s+x+1] = row+integral[y*s+x+1];
        }
    }
    for (size_t y=0; y<h; ++y) for (size_t x=0; x<w; ++x) {
        size_t l=x>r?x-r:0, t=y>r?y-r:0;
        size_t right=std::min(w,x+r+1), bottom=std::min(h,y+r+1);
        out[y*w+x]=(integral[bottom*s+right]-integral[t*s+right]
                     -integral[bottom*s+l]+integral[t*s+l])/double((right-l)*(bottom-t));
    }
    return out;
}

inline void guided(const float* src, float* dst, size_t w, size_t h, size_t r, double scale) {
    // Self-guided filter: I=p. epsilon=scale^2, not the scale itself.
    std::vector<double> pixels(src,src+w*h), squared(w*h);
    for (size_t i=0;i<w*h;++i) squared[i]=pixels[i]*pixels[i];
    auto mean=box(pixels,w,h,r), second=box(squared,w,h,r);
    for (size_t i=0;i<w*h;++i) {
        double variance=std::max(0.,second[i]-mean[i]*mean[i]);
        squared[i]=variance/(variance+scale*scale); // a
        second[i]=(1-squared[i])*mean[i];          // b
    }
    auto a=box(squared,w,h,r), b=box(second,w,h,r);
    for (size_t i=0;i<w*h;++i) dst[i]=float(std::clamp(a[i]*pixels[i]+b[i],0.,1.));
}

// Monotonic queues: each index enters/exits at most once per axis.
inline void morphology(const float* src, float* dst, size_t w, size_t h, size_t r, bool maximum) {
    std::vector<float> temporary(w*h);
    std::vector<size_t> queue(std::max(w,h));
    auto axis=[&](const float* input, float* output, size_t lines, size_t length,
                  size_t line_stride, size_t step) {
        for (size_t line=0;line<lines;++line) {
            size_t head=0,tail=0,next=0,base=line*line_stride;
            for (size_t x=0;x<length;++x) {
                size_t left=x>r?x-r:0, right=std::min(length,x+r+1);
                while(head<tail && queue[head]<left) ++head;
                while(next<right) {
                    float value=input[base+next*step];
                    while(head<tail && (maximum ? input[base+queue[tail-1]*step]<=value
                                                         : input[base+queue[tail-1]*step]>=value)) --tail;
                    queue[tail++]=next++;
                }
                output[base+x*step]=input[base+queue[head]*step];
            }
        }
    };
    axis(src,temporary.data(),h,w,w,1);
    axis(temporary.data(),dst,w,h,1,w);
}

struct Blend { size_t lo, hi; double weight; };
inline std::vector<Blend> blends(size_t length,size_t tile) {
    std::vector<double> centers;
    for(size_t start=0;start<length;start+=tile)
        centers.push_back((double(start)+double(std::min(length,start+tile))-1)*0.5);
    std::vector<Blend> result(length);
    size_t hi=0;
    for(size_t x=0;x<length;++x) {
        while(hi<centers.size() && centers[hi]<double(x)) ++hi;
        if(hi==0) result[x]={0,0,0};
        else if(hi==centers.size()) result[x]={hi-1,hi-1,0};
        else result[x]={hi-1,hi,(x-centers[hi-1])/(centers[hi]-centers[hi-1])};
    }
    return result;
}

inline void clahe(const float* src,float* dst,size_t w,size_t h,size_t tile,double clip) {
    // 256 bins; fractional uniform redistribution. No padding of partial tiles.
    const size_t nx=(w-1)/tile+1, ny=(h-1)/tile+1;
    std::vector<std::array<float,256>> maps(nx*ny);
    auto bin=[](float v){return size_t(double(v)*255+0.5);};
    for(size_t ty=0;ty<ny;++ty) for(size_t tx=0;tx<nx;++tx) {
        size_t x0=tx*tile,y0=ty*tile,x1=std::min(w,x0+tile),y1=std::min(h,y0+tile);
        double count=double((x1-x0)*(y1-y0));
        std::array<double,256> histogram{};
        for(size_t y=y0;y<y1;++y) for(size_t x=x0;x<x1;++x) ++histogram[bin(src[y*w+x])];
        double limit=std::max(1.,clip*count/256.), excess=0;
        for(auto& value:histogram) { excess+=std::max(0.,value-limit); value=std::min(value,limit); }
        double cumulative=0;
        for(size_t k=0;k<256;++k) {
            cumulative+=histogram[k]+excess/256.;
            maps[ty*nx+tx][k]=float(cumulative/count);
        }
    }
    auto bx=blends(w,tile), by=blends(h,tile);
    for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x) {
        const auto xx=bx[x], yy=by[y]; size_t k=bin(src[y*w+x]);
        double top=maps[yy.lo*nx+xx.lo][k]*(1-xx.weight)+maps[yy.lo*nx+xx.hi][k]*xx.weight;
        double bottom=maps[yy.hi*nx+xx.lo][k]*(1-xx.weight)+maps[yy.hi*nx+xx.hi][k]*xx.weight;
        dst[y*w+x]=float(std::clamp(top*(1-yy.weight)+bottom*yy.weight,0.,1.));
    }
}
} // namespace advanced
