#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Squared distances deliberately use the same separate operations as NumPy.
// NaN validation is part of the ABI; do not compile with fast-math.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
namespace nn_detail {
constexpr size_t max_points = 10000000;
constexpr double max_coordinate = 1e150; // Keeps squared 3D differences finite.
inline bool aligned(const void* p) { return reinterpret_cast<uintptr_t>(p)%8==0; }
inline bool overlap(const void* a,size_t na,const void* b,size_t nb) {
    if(!na || !nb) return false;
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x>=y ? x-y<nb : y-x<na;
}
inline bool points_ok(const double* p,size_t n) {
    if(n>max_points || (n && !p) || !aligned(p)) return false;
    for(size_t i=0;i<3*n;++i)
        if(!std::isfinite(p[i]) || std::abs(p[i])>max_coordinate) return false;
    return true;
}
struct Node { double p[3]; int64_t index; unsigned axis; };
struct Tree {
    std::vector<Node> nodes;
    Tree(const double* points,size_t n):nodes(n) {
        for(size_t i=0;i<n;++i) {
            for(unsigned a=0;a<3;++a) nodes[i].p[a]=points[3*i+a];
            nodes[i].index=int64_t(i);
        }
        build(0,n);
    }
    void build(size_t lo,size_t hi) {
        if(lo>=hi) return;
        double low[3],high[3];
        for(unsigned a=0;a<3;++a) low[a]=high[a]=nodes[lo].p[a];
        for(size_t i=lo+1;i<hi;++i) for(unsigned a=0;a<3;++a) {
            low[a]=std::min(low[a],nodes[i].p[a]);
            high[a]=std::max(high[a],nodes[i].p[a]);
        }
        unsigned axis=0;
        for(unsigned a=1;a<3;++a) if(high[a]-low[a]>high[axis]-low[axis]) axis=a;
        const size_t mid=lo+(hi-lo)/2;
        std::nth_element(nodes.begin()+lo,nodes.begin()+mid,nodes.begin()+hi,
            [axis](const Node& a,const Node& b) {
                return a.p[axis]<b.p[axis] || (a.p[axis]==b.p[axis] && a.index<b.index);
            });
        nodes[mid].axis=axis;
        build(lo,mid);build(mid+1,hi);
    }
    void nearest(const double* q,size_t lo,size_t hi,double& best,int64_t& index) const {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        if(lo>=hi) return;
        const size_t mid=lo+(hi-lo)/2;
        const Node& node=nodes[mid];
        const double x=q[0]-node.p[0],y=q[1]-node.p[1],z=q[2]-node.p[2];
        const double distance=(x*x+y*y)+z*z;
        if(distance<best || (distance==best && (index<0 || node.index<index))) {
            best=distance;index=node.index;
        }
        const double delta=q[node.axis]-node.p[node.axis];
        if(delta<=0) {
            nearest(q,lo,mid,best,index);
            if(delta*delta<=best) nearest(q,mid+1,hi,best,index);
        } else {
            nearest(q,mid+1,hi,best,index);
            if(delta*delta<=best) nearest(q,lo,mid,best,index);
        }
    }
};
inline std::mutex registry_mutex;
inline std::unordered_map<uint64_t,std::shared_ptr<const Tree>> registry;
inline uint64_t next_handle=1;
}

// Inputs/outputs must have declared extents. Return 0 success, 1 bad input,
// 2 allocation/other exception, 3 unknown/released handle. All validation
// finishes before outputs are changed. Handle IDs are never reused.
extern "C" int nn3_create(const double* points,size_t n,uint64_t* handle) noexcept {
    if(!n || !handle || !nn_detail::aligned(handle) || !nn_detail::points_ok(points,n) ||
       nn_detail::overlap(points,n*24,handle,8)) return 1;
    try {
        auto tree=std::make_shared<nn_detail::Tree>(points,n);
        std::lock_guard<std::mutex> lock(nn_detail::registry_mutex);
        if(nn_detail::next_handle==0) return 2;
        const auto id=nn_detail::next_handle++;
        nn_detail::registry.emplace(id,std::move(tree));
        *handle=id;
        return 0;
    } catch(...) { return 2; }
}
extern "C" int nn3_query(uint64_t handle,const double* query,size_t n,double max_distance,
                          int64_t* indices,double* squared_distances) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if(std::isnan(max_distance) || max_distance<0 || !nn_detail::points_ok(query,n) ||
       (n && (!indices || !squared_distances)) || !nn_detail::aligned(indices) ||
       !nn_detail::aligned(squared_distances) ||
       nn_detail::overlap(query,n*24,indices,n*8) ||
       nn_detail::overlap(query,n*24,squared_distances,n*8) ||
       nn_detail::overlap(indices,n*8,squared_distances,n*8)) return 1;
    try {
        std::shared_ptr<const nn_detail::Tree> tree;
        {
            std::lock_guard<std::mutex> lock(nn_detail::registry_mutex);
            const auto found=nn_detail::registry.find(handle);
            if(found==nn_detail::registry.end()) return 3;
            tree=found->second;
        }
        const double limit=max_distance>1e154 ? std::numeric_limits<double>::infinity()
                                             : max_distance*max_distance;
        for(size_t i=0;i<n;++i) {
            double best=limit;int64_t index=-1;
            tree->nearest(query+3*i,0,tree->nodes.size(),best,index);
            indices[i]=index;
            squared_distances[i]=index<0 ? std::numeric_limits<double>::infinity() : best;
        }
        return 0;
    } catch(...) { return 2; }
}
extern "C" int nn3_release(uint64_t handle) noexcept {
    try {
        std::lock_guard<std::mutex> lock(nn_detail::registry_mutex);
        return nn_detail::registry.erase(handle) ? 0 : 3;
    } catch(...) { return 2; }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
