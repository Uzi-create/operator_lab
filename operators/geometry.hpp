#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace geometry {
// Lower envelope of parabolas f[q]+(x-q)^2. Infinite sites are skipped.
inline void edt_line(const double* input, double* output, size_t length,
                     std::vector<size_t>& sites, std::vector<double>& breaks) {
    const double inf=std::numeric_limits<double>::infinity();
    size_t count=0;
    for(size_t q=0;q<length;++q) {
        if(!std::isfinite(input[q])) continue;
        double crossing=-inf;
        while(count) {
            size_t v=sites[count-1];
            crossing=((input[q]+double(q)*q)-(input[v]+double(v)*v))/(2.*double(q-v));
            if(crossing>breaks[count-1]) break;
            --count;
        }
        sites[count]=q;
        breaks[count]=count?crossing:-inf;
        ++count;
    }
    if(!count) { std::fill(output,output+length,inf); return; }
    size_t k=0;
    for(size_t x=0;x<length;++x) {
        while(k+1<count && breaks[k+1]<double(x)) ++k;
        double delta=double(x)-double(sites[k]);
        output[x]=delta*delta+input[sites[k]];
    }
}

// Exact squared Euclidean distance to the selected class of pixel centers.
inline std::vector<double> distance(const float* src,size_t w,size_t h,bool to_foreground) {
    const double inf=std::numeric_limits<double>::infinity();
    size_t length=std::max(w,h);
    std::vector<double> result(w*h),input(length),output(length),breaks(length);
    std::vector<size_t> sites(length);
    for(size_t y=0;y<h;++y) {
        for(size_t x=0;x<w;++x)
            input[x]=((src[y*w+x]>0.5f)==to_foreground)?0.:inf;
        edt_line(input.data(),output.data(),w,sites,breaks);
        std::copy(output.begin(),output.begin()+w,result.begin()+y*w);
    }
    for(size_t x=0;x<w;++x) {
        for(size_t y=0;y<h;++y) input[y]=result[y*w+x];
        edt_line(input.data(),output.data(),h,sites,breaks);
        for(size_t y=0;y<h;++y) result[y*w+x]=output[y];
    }
    return result;
}

inline void distance_output(const float* src,float* dst,size_t w,size_t h,int mode,double feather_width) {
    auto to_background=distance(src,w,h,false);
    if(mode==1) {
        for(size_t i=0;i<w*h;++i) dst[i]=float(std::sqrt(to_background[i]));
        return;
    }
    auto to_foreground=distance(src,w,h,true);
    for(size_t i=0;i<w*h;++i) {
        const double signed_distance=src[i]>0.5f?std::sqrt(to_background[i]):-std::sqrt(to_foreground[i]);
        if(mode==2) dst[i]=float(signed_distance);
        else {
            const double t=std::clamp(0.5+signed_distance/(2.*feather_width),0.,1.);
            dst[i]=float(t*t*(3.-2.*t));
        }
    }
}

inline size_t clamp_index(long long x,size_t limit) {
    return size_t(std::max(0LL,std::min(x,static_cast<long long>(limit)-1)));
}

inline void hysteresis(const std::vector<double>& suppressed,float* dst,size_t w,size_t h,
                       double low,double high) {
    std::fill(dst,dst+w*h,0.f);
    std::vector<size_t> stack;
    for(size_t i=0;i<w*h;++i) if(suppressed[i]>=high) {
        dst[i]=1.f; stack.push_back(i);
    }
    while(!stack.empty()) {
        size_t i=stack.back(); stack.pop_back();
        const size_t x=i%w,y=i/w;
        for(size_t yy=y?y-1:0;yy<std::min(h,y+2);++yy)
            for(size_t xx=x?x-1:0;xx<std::min(w,x+2);++xx) {
                const size_t j=yy*w+xx;
                if(dst[j]==0.f && suppressed[j]>=low) { dst[j]=1.f; stack.push_back(j); }
            }
    }
}

inline void canny(const float* src,float* dst,size_t w,size_t h,double low,double high) {
    // Fixed separable 5-tap binomial smoothing; replicate borders.
    constexpr double kernel[5]={1./16,4./16,6./16,4./16,1./16};
    std::vector<double> tmp(w*h),blur(w*h),magnitude(w*h),gx(w*h),gy(w*h),suppressed(w*h,0);
    for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x)
        for(int k=-2;k<=2;++k) tmp[y*w+x]+=kernel[k+2]*src[y*w+clamp_index(static_cast<long long>(x)+k,w)];
    for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x)
        for(int k=-2;k<=2;++k) blur[y*w+x]+=kernel[k+2]*tmp[clamp_index(static_cast<long long>(y)+k,h)*w+x];
    for(size_t y=1;y+1<h;++y) for(size_t x=1;x+1<w;++x) {
        size_t i=y*w+x;
        gx[i]=(blur[i-w+1]+2*blur[i+1]+blur[i+w+1]-blur[i-w-1]-2*blur[i-1]-blur[i+w-1])/4.;
        gy[i]=(blur[i+w-1]+2*blur[i+w]+blur[i+w+1]-blur[i-w-1]-2*blur[i-w]-blur[i-w+1])/4.;
        magnitude[i]=std::hypot(gx[i],gy[i]);
    }
    constexpr double tan22=0.4142135623730950488, tan67=2.4142135623730950488;
    for(size_t y=1;y+1<h;++y) for(size_t x=1;x+1<w;++x) {
        size_t i=y*w+x,first,second;
        double ax=std::abs(gx[i]),ay=std::abs(gy[i]);
        if(ay<=tan22*ax) { first=i-1; second=i+1; }
        else if(ay>=tan67*ax) { first=i-w; second=i+w; }
        else if(gx[i]*gy[i]>0) { first=i-w-1; second=i+w+1; }
        else { first=i-w+1; second=i+w-1; }
        // Asymmetric comparison resolves flat ridges to one side deterministically.
        if(magnitude[i]>magnitude[first] && magnitude[i]>=magnitude[second]) suppressed[i]=magnitude[i];
    }
    hysteresis(suppressed,dst,w,h,low,high);
}
} // namespace geometry

extern "C" int run_geometry(const float* src,float* dst,int width,int height,
                             int mode,float first,float second) noexcept {
    if(!src || !dst || src==dst || width<=0 || height<=0 || mode<0 || mode>3 ||
       !std::isfinite(first) || !std::isfinite(second) ||
       (mode==0 && (first<=0 || second<first)) || (mode==3 && first<=0)) return 1;
    const size_t w=width,h=height;
    if(w>std::numeric_limits<size_t>::max()/h) return 1;
    for(size_t i=0;i<w*h;++i) if(!std::isfinite(src[i]) || src[i]<0 || src[i]>1) return 1;
    try {
        if(mode==0) geometry::canny(src,dst,w,h,first,second);
        else geometry::distance_output(src,dst,w,h,mode,first);
        return 0;
    } catch(...) { return 2; }
}
