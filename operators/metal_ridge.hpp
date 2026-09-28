#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ridge_detail {
inline float pair(float center,float first,float second) {
    const float a=center-first,b=center-second;
    return std::max(std::min(a,b),std::min(-a,-b));
}
#if defined(_MSC_VER)
#define RIDGE_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RIDGE_RESTRICT __restrict__
#else
#define RIDGE_RESTRICT
#endif
// Restrict is valid because the exported entry point rejects buffer overlap.
inline void kernel(const float* RIDGE_RESTRICT src,float* RIDGE_RESTRICT dst,size_t w,size_t h) {
    for(size_t y=0;y<h;++y) {
        const float* c=src+y*w;
        const float* m1=src+(y>0?y-1:0)*w;
        const float* m2=src+(y>1?y-2:0)*w;
        const float* m3=src+(y>2?y-3:0)*w;
        const float* p1=src+std::min(h-1,y+1)*w;
        const float* p2=src+std::min(h-1,y+2)*w;
        const float* p3=src+std::min(h-1,y+3)*w;
        auto edge=[&](size_t x) {
            float best=0;
            const int dxs[]={1,0,1,1},dys[]={0,1,1,-1};
            for(int d=0;d<4;++d) for(int r=1;r<=3;++r) {
                const auto clamp=[](long long q,size_t size) {return size_t(std::clamp(q,0LL,static_cast<long long>(size)-1));};
                const size_t xa=clamp(static_cast<long long>(x)+dxs[d]*r,w);
                const size_t xb=clamp(static_cast<long long>(x)-dxs[d]*r,w);
                const size_t ya=clamp(static_cast<long long>(y)+dys[d]*r,h);
                const size_t yb=clamp(static_cast<long long>(y)-dys[d]*r,h);
                best=std::max(best,pair(c[x],src[ya*w+xa],src[yb*w+xb]));
            }
            dst[y*w+x]=best;
        };
        const size_t left=std::min<size_t>(3,w),right=w>6?w-3:left;
        for(size_t x=0;x<left;++x) edge(x);
        // Fuse twelve responses, write each output once. Constant offsets expose SIMD.
        for(size_t x=left;x<right;++x) {
            const float value=c[x];float best=0;
            best=std::max(best,pair(value,c[x+1],c[x-1]));
            best=std::max(best,pair(value,c[x+2],c[x-2]));
            best=std::max(best,pair(value,c[x+3],c[x-3]));
            best=std::max(best,pair(value,p1[x],m1[x]));
            best=std::max(best,pair(value,p2[x],m2[x]));
            best=std::max(best,pair(value,p3[x],m3[x]));
            best=std::max(best,pair(value,p1[x+1],m1[x-1]));
            best=std::max(best,pair(value,p2[x+2],m2[x-2]));
            best=std::max(best,pair(value,p3[x+3],m3[x-3]));
            best=std::max(best,pair(value,m1[x+1],p1[x-1]));
            best=std::max(best,pair(value,m2[x+2],p2[x-2]));
            best=std::max(best,pair(value,m3[x+3],p3[x-3]));
            dst[y*w+x]=best;
        }
        for(size_t x=right;x<w;++x) edge(x);
    }
}
#undef RIDGE_RESTRICT
}

// Replicate borders. Invalid input never modifies output. Do not use fast-math:
// finite [0,1] validation is part of this ABI. Caller owns sufficient buffers.
extern "C" int metal_ridge(const float* src,float* dst,int width,int height) noexcept {
    if(!src || !dst || width<=0 || height<=0) return 1;
    const size_t w=width,h=height;
    if(w>std::numeric_limits<size_t>::max()/h) return 1;
    const size_t n=w*h;
    if(n>std::numeric_limits<size_t>::max()/sizeof(float)) return 1;
    const auto a=reinterpret_cast<uintptr_t>(src),b=reinterpret_cast<uintptr_t>(dst);
    if((a>=b?a-b:b-a)<n*sizeof(float)) return 1;
    bool invalid=false;
    for(size_t i=0;i<n;++i) invalid |= !(src[i]>=0.f && src[i]<=1.f);
    if(invalid) return 1;
    ridge_detail::kernel(src,dst,w,h);
    return 0;
}
