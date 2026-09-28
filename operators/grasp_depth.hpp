#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace grasp_depth {
inline double median(std::vector<double> v) {
    size_t k=v.size()/2;
    std::nth_element(v.begin(),v.begin()+k,v.end());
    double upper=v[k];
    return v.size()%2?upper:(upper+*std::max_element(v.begin(),v.begin()+k))/2.;
}
struct Sample { double z; size_t index; };
// Largest interval whose total depth span is <= band. Equal counts favor nearer depth.
inline std::pair<size_t,size_t> densest(const std::vector<Sample>& v,double band) {
    size_t left=0,best_left=0,best_right=0;
    for(size_t right=0;right<v.size();++right) {
        while(v[right].z-v[left].z>band) ++left;
        if(right-left+1>best_right-best_left) { best_left=left; best_right=right+1; }
    }
    return {best_left,best_right};
}
}

// Results: area, valid, cluster, valid_fraction, cluster_fraction, runner_ratio,
// median_z, MAD_z, median_u, median_v, median_X, median_Y, bbox_width_m, bbox_height_m.
// Status: 0 accepted, 1 invalid args, 2 allocation failure, 3 insufficient support,
// 4 ambiguous depth surfaces, 5 excessive MAD. Outputs always initialize to zero.
extern "C" int estimate_grasp_depth(const float* depth,const unsigned char* mask,
 int width,int height,const double* config,double* result,unsigned char* inliers) noexcept {
    if(!depth || !mask || !config || !result || !inliers || width<=0 || height<=0) return 1;
    const size_t w=width,h=height;
    if(w>std::numeric_limits<size_t>::max()/h) return 1;
    // config: min_z,max_z,full_band,min_samples,min_valid_fraction,
    // min_cluster_fraction,max_MAD,ambiguity_ratio,fx,fy,cx,cy.
    for(size_t j=0;j<12;++j) if(!std::isfinite(config[j])) return 1;
    const double lo=config[0],hi=config[1],band=config[2];
    if(lo<=0 || hi<=lo || band<=0 || config[3]<1 || config[3]!=std::floor(config[3]) ||
       config[4]<0 || config[4]>1 || config[5]<=0 || config[5]>1 || config[6]<0 ||
       config[7]<=0 || config[7]>1 || config[8]<=0 || config[9]<=0) return 1;
    std::fill(result,result+14,0.); std::fill(inliers,inliers+w*h,0);
    try {
        std::vector<grasp_depth::Sample> samples;
        size_t xmin=w,ymin=h,xmax=0,ymax=0,area=0;
        for(size_t i=0;i<w*h;++i) if(mask[i]) {
            ++area; xmin=std::min(xmin,i%w);xmax=std::max(xmax,i%w);
            ymin=std::min(ymin,i/w);ymax=std::max(ymax,i/w);
            if(std::isfinite(depth[i]) && depth[i]>=lo && depth[i]<=hi) samples.push_back({depth[i],i});
        }
        result[0]=double(area);result[1]=double(samples.size());
        result[3]=area?double(samples.size())/area:0;
        if(double(samples.size())<config[3] || result[3]<config[4]) return 3;
        std::sort(samples.begin(),samples.end(),[](const auto& a,const auto& b){
            return a.z<b.z || (a.z==b.z && a.index<b.index);
        });
        auto bounds=grasp_depth::densest(samples,band);
        const size_t count=bounds.second-bounds.first;
        result[2]=double(count);result[4]=double(count)/samples.size();
        std::vector<grasp_depth::Sample> remainder;
        for(size_t j=0;j<samples.size();++j)
            if(j<bounds.first || j>=bounds.second) remainder.push_back(samples[j]);
        auto runner=grasp_depth::densest(remainder,band);
        result[5]=double(runner.second-runner.first)/count;
        if(double(count)<config[3] || result[4]<config[5]) return 3;
        if(result[5]>=config[7]) return 4;
        std::vector<double> zs,us,vs,xs,ys;
        for(size_t j=bounds.first;j<bounds.second;++j) {
            const auto s=samples[j];const double u=double(s.index%w),v=double(s.index/w);
            zs.push_back(s.z);us.push_back(u);vs.push_back(v);
            xs.push_back((u-config[10])*s.z/config[8]);ys.push_back((v-config[11])*s.z/config[9]);
        }
        result[6]=grasp_depth::median(zs);
        for(auto& z:zs) z=std::abs(z-result[6]);
        result[7]=grasp_depth::median(zs);
        if(result[7]>config[6]) return 5;
        result[8]=grasp_depth::median(us);result[9]=grasp_depth::median(vs);
        result[10]=grasp_depth::median(xs);result[11]=grasp_depth::median(ys);
        result[12]=double(xmax-xmin+1)*result[6]/config[8];
        result[13]=double(ymax-ymin+1)*result[6]/config[9];
        for(size_t j=bounds.first;j<bounds.second;++j) inliers[samples[j].index]=1;
        return 0;
    } catch(...) { return 2; }
}
