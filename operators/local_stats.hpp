#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Independent implementation of masked integral moments. One pass builds three
// tables; clipped square/ring queries are O(1). A workspace belongs to one caller.
struct StatsWorkspace {
    std::vector<double> sum, square;
    std::vector<int32_t> count;
};
extern "C" void* stats_create() noexcept {
    try { return new StatsWorkspace; } catch (...) { return nullptr; }
}
extern "C" void stats_destroy(void* workspace) noexcept {
    delete static_cast<StatsWorkspace*>(workspace);
}
extern "C" int masked_stats(void* workspace, const double* src, const uint8_t* mask,
    int width, int height, int inner, int outer, double noise, int minimum,
    double fraction, float* background, float* variance, float* residual,
    float* score, uint8_t* valid, int32_t* counts, int32_t* capacities) noexcept {
    if (!workspace || !src || !mask || !background || !variance || !residual ||
        !score || !valid || !counts || !capacities || width<=0 || height<=0 ||
        inner < -1 || outer<0 || inner>=outer || outer>4096 || minimum<1 ||
        !std::isfinite(noise) || noise<1e-6 || noise>1 ||
        !std::isfinite(fraction) || fraction<=0 || fraction>1) return 1;
    const size_t w=width,h=height,s=w+1;
    if (w>size_t(INT32_MAX)/h || s>std::numeric_limits<size_t>::max()/(h+1)) return 1;
    for(size_t i=0;i<w*h;++i)
        if(!std::isfinite(src[i]) || src[i]<0 || src[i]>1) return 1;
    try {
        auto& a=*static_cast<StatsWorkspace*>(workspace);
        a.sum.resize(s*(h+1)); a.square.resize(s*(h+1)); a.count.resize(s*(h+1));
        std::fill_n(a.sum.begin(),s,0.); std::fill_n(a.square.begin(),s,0.);
        std::fill_n(a.count.begin(),s,0);
        for(size_t y=0;y<h;++y) {
            double sum=0,square=0; int32_t count=0;
            a.sum[(y+1)*s]=0; a.square[(y+1)*s]=0; a.count[(y+1)*s]=0;
            for(size_t x=0;x<w;++x) {
                const size_t i=y*w+x,j=(y+1)*s+x+1;
                if(mask[i]) { const double v=src[i]; sum+=v;square+=v*v;++count; }
                a.sum[j]=a.sum[j-s]+sum;a.square[j]=a.square[j-s]+square;
                a.count[j]=a.count[j-s]+count;
            }
        }
        struct Moment { double sum,square; int32_t count,area; };
        auto box=[&](size_t x,size_t y,int radius) -> Moment {
            if(radius<0) return {0,0,0,0};
            const size_t r=radius,l=x>r?x-r:0,t=y>r?y-r:0;
            const size_t right=std::min(w,x+r+1),bottom=std::min(h,y+r+1);
            const size_t aa=bottom*s+right,bb=t*s+right,cc=bottom*s+l,dd=t*s+l;
            return {a.sum[aa]-a.sum[bb]-a.sum[cc]+a.sum[dd],
                    a.square[aa]-a.square[bb]-a.square[cc]+a.square[dd],
                    a.count[aa]-a.count[bb]-a.count[cc]+a.count[dd],
                    int32_t((right-l)*(bottom-t))};
        };
        for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x) {
            const size_t i=y*w+x; const auto out=box(x,y,outer),in=box(x,y,inner);
            const int32_t count=out.count-in.count,area=out.area-in.area;
            counts[i]=count;capacities[i]=area;
            const bool ok=mask[i] && count>=minimum && area>0 && count>=area*fraction;
            valid[i]=uint8_t(ok);
            if(!ok) {background[i]=variance[i]=residual[i]=score[i]=0;continue;}
            const double mean=(out.sum-in.sum)/count;
            const double var=std::max(0.,(out.square-in.square)/count-mean*mean);
            const double delta=src[i]-mean;
            background[i]=float(mean);variance[i]=float(var);residual[i]=float(delta);
            score[i]=float(std::abs(delta)/std::sqrt(var+noise*noise));
        }
        return 0;
    } catch (...) { return 2; }
}
