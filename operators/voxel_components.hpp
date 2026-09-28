#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace voxel_detail {
using Key=std::array<int64_t,3>;
struct Hasher {
    size_t operator()(const Key& key) const noexcept {
        size_t value=0x9e3779b97f4a7c15ULL;
        for(int64_t item:key) value^=std::hash<int64_t>{}(item)+0x9e3779b97f4a7c15ULL+
                                      (value<<6)+(value>>2);
        return value;
    }
};
template<class T> bool aligned(const T* pointer) {
    return reinterpret_cast<uintptr_t>(pointer)%alignof(T)==0;
}
inline bool overlap(const void* a,size_t na,const void* b,size_t nb) {
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x>=y ? x-y<nb : y-x<na;
}
}

// One label per UNIQUE voxel key. Input order determines stable component
// labels (the Python caller supplies lexicographically sorted unique keys).
// 0 success; 1 bad arguments/duplicate key; 2 allocation/internal failure.
// Output arrays remain unchanged on bad input.
extern "C" int voxel_connected_components(const int64_t* keys,size_t n,int connectivity,
                                            size_t min_voxels,int32_t* labels,
                                            int32_t* cluster_count) noexcept {
    if(!keys || !labels || !cluster_count || !n || n>2000000 || !min_voxels ||
       min_voxels>n || (connectivity!=6 && connectivity!=18 && connectivity!=26) ||
       !voxel_detail::aligned(keys) || !voxel_detail::aligned(labels) ||
       !voxel_detail::aligned(cluster_count) ||
       voxel_detail::overlap(keys,n*24,labels,n*4) ||
       voxel_detail::overlap(keys,n*24,cluster_count,4) ||
       voxel_detail::overlap(labels,n*4,cluster_count,4)) return 1;
    try {
        using voxel_detail::Key;
        std::unordered_map<Key,size_t,voxel_detail::Hasher> lookup;
        lookup.reserve(n*2);
        constexpr int64_t limit=9000000000000000000LL;
        for(size_t i=0;i<n;++i) {
            const Key key={keys[3*i],keys[3*i+1],keys[3*i+2]};
            for(auto value:key) if(value < -limit || value > limit) return 1;
            if(!lookup.emplace(key,i).second) return 1;
        }
        std::vector<size_t> parent(n),sizes(n,1);
        std::iota(parent.begin(),parent.end(),size_t(0));
        auto root=[&parent](size_t i) {
            while(parent[i]!=i) {
                parent[i]=parent[parent[i]];
                i=parent[i];
            }
            return i;
        };
        for(size_t i=0;i<n;++i) {
            const Key key={keys[3*i],keys[3*i+1],keys[3*i+2]};
            for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) for(int dz=-1;dz<=1;++dz) {
                const int manhattan=(dx<0?-dx:dx)+(dy<0?-dy:dy)+(dz<0?-dz:dz);
                if(!manhattan || (connectivity==6 && manhattan>1) ||
                   (connectivity==18 && manhattan>2)) continue;
                const Key neighbor={key[0]+dx,key[1]+dy,key[2]+dz};
                const auto found=lookup.find(neighbor);
                if(found==lookup.end() || found->second<=i) continue;
                size_t a=root(i),b=root(found->second);
                if(a==b) continue;
                if(a>b) std::swap(a,b);
                parent[b]=a;sizes[a]+=sizes[b];
            }
        }
        std::vector<int32_t> root_label(n,0),temporary(n,0);
        int32_t total=0;
        for(size_t i=0;i<n;++i) {
            const size_t r=root(i);
            if(sizes[r]<min_voxels) continue;
            if(!root_label[r]) root_label[r]=++total;
            temporary[i]=root_label[r];
        }
        for(size_t i=0;i<n;++i) labels[i]=temporary[i];
        *cluster_count=total;
        return 0;
    } catch(...) { return 2; }
}
