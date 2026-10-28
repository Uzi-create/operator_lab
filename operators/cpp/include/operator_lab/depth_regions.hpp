#pragma once
#include "rgbd_geometry.hpp"
#include <algorithm>
#include <functional>
#include <optional>
#include <utility>

namespace operator_lab::rgbd {

class MaskView {
public:
    MaskView(const std::uint8_t* data, std::size_t elements, std::size_t width,
             std::size_t height, std::size_t stride = 0)
        : data_(data), width_(width), height_(height), stride_(stride ? stride : width) {
        if (!data || !width || !height || width > 16384 || height > 16384 ||
            width > 16000000 / height || stride_ < width || elements < width ||
            height - 1 > (elements - width) / stride_)
            throw std::invalid_argument("invalid mask buffer or stride");
    }
    std::size_t width() const { return width_; }
    std::size_t height() const { return height_; }
    bool contains(std::size_t u, std::size_t v) const {
        if (u >= width_ || v >= height_) throw std::out_of_range("mask pixel outside image");
        return data_[v * stride_ + u] != 0;
    }
private:
    const std::uint8_t* data_;
    std::size_t width_, height_, stride_;
};

struct DepthRegionConfig {
    double min_depth_m = 0.2, max_depth_m = 8.;
    double absolute_jump_m = 0.08, relative_jump = 0.02;
    std::size_t min_pixels = 12;
    int connectivity = 4;
};
struct DepthRegion {
    std::size_t label = 0; // Local to this frame, never a persistent identity.
    Rect bounds;
    std::vector<Pixel> pixels;
    double depth_m = 0, mad_m = 0;
    Pixel representative_pixel;
    Point3 point_m; // Actual measured depth at representative_pixel.
};
inline void validate(const DepthRegionConfig& c) {
    if (!std::isfinite(c.min_depth_m) || c.min_depth_m < 0 ||
        !std::isfinite(c.max_depth_m) || c.max_depth_m <= c.min_depth_m ||
        !std::isfinite(c.absolute_jump_m) || c.absolute_jump_m < 0 ||
        !std::isfinite(c.relative_jump) || c.relative_jump < 0 ||
        !c.min_pixels || (c.connectivity != 4 && c.connectivity != 8))
        throw std::invalid_argument("invalid depth region configuration");
}
namespace region_detail {
struct WeightedValue {double value;std::uint32_t count;};
inline bool connected(const std::function<bool(Pixel,Pixel)>& fn,Pixel a,Pixel b){return !fn||fn(a,b);}
inline bool connected(std::nullptr_t,Pixel,Pixel){return true;}
template<class Connector> inline bool connected(const Connector& fn,Pixel a,Pixel b){
    if constexpr(std::is_pointer_v<Connector>)return !fn||fn(a,b);
    else return fn(a,b);
}
inline double metric_code(std::uint16_t code,double scale) {
    // Force the same stored double rounding as the original per-pixel buffer,
    // including compilers which contract multiply/subtract across statements.
    volatile double metric=static_cast<double>(code)*scale;
    return metric;
}
inline double median_inplace(std::vector<double>& values) {
    const auto n = values.size();
    if(std::all_of(values.begin()+1,values.end(),[&](double value){return value==values.front();}))
        return n%2?values.front():values.front()/2+values.front()/2;
    auto middle = values.begin() + static_cast<std::ptrdiff_t>(n / 2);
    if(n>=4096){
        std::sort(values.begin(),values.end());
        return n%2?*middle:*(middle-1)/2+*middle/2;
    }
    std::nth_element(values.begin(), middle, values.end());
    return n % 2 ? *middle : *std::max_element(values.begin(), middle) / 2 + *middle / 2;
}
}

// Optional reusable scratch; retains high-water capacities, never image views or
// returned regions. One workspace per concurrent caller; clear() releases storage.
struct DepthRegionWorkspace {
    std::vector<double> depth_m,values,scratch;
    std::vector<std::uint8_t> visited;
    std::vector<std::size_t> queue;
    std::vector<std::uint32_t> histogram;
    std::vector<std::uint16_t> active_bins;
    std::vector<region_detail::WeightedValue> weighted;
    void clear(){DepthRegionWorkspace empty;swap(empty);}
    void swap(DepthRegionWorkspace& other) noexcept {
        depth_m.swap(other.depth_m);values.swap(other.values);scratch.swap(other.scratch);
        visited.swap(other.visited);queue.swap(other.queue);
        histogram.swap(other.histogram);active_bins.swap(other.active_bins);weighted.swap(other.weighted);
    }
};

// Local depth-jump connectivity is transitive: a ramp can form one component.
// No person detection, semantic ground removal or automatic RGB registration.
template<class T,class Connector=std::function<bool(Pixel,Pixel)>>
inline std::vector<DepthRegion> extract_depth_regions(const DepthView<T>& depth,
    const Intrinsics& k, Rect roi, const DepthRegionConfig& config = {},
    std::optional<MaskView> mask = std::nullopt,
    const Connector& can_connect = {},
    DepthRegionWorkspace* workspace = nullptr) {
    validate(k); validate(config); validate(roi, depth.width(), depth.height());
    if (k.width != depth.width() || k.height != depth.height() ||
        (mask && (mask->width() != depth.width() || mask->height() != depth.height())))
        throw std::invalid_argument("region input dimensions differ");
    // Preserve the original std::function acceptance of mutable callables while
    // allowing the tracker's const RGB-boundary lambda to inline without erasure.
    std::function<bool(Pixel,Pixel)> mutable_connector;
    if constexpr(!std::is_invocable_r_v<bool,const Connector&,Pixel,Pixel> && !std::is_null_pointer_v<Connector>)
        mutable_connector=can_connect;
    const auto connected=[&](Pixel a,Pixel b){
        if constexpr(std::is_invocable_r_v<bool,const Connector&,Pixel,Pixel> || std::is_null_pointer_v<Connector>)
            return region_detail::connected(can_connect,a,b);
        else return mutable_connector(a,b);
    };
    const std::size_t n = roi.width * roi.height;
    DepthRegionWorkspace local;
    auto& work = workspace ? *workspace : local;
    auto& z = work.depth_m; z.assign(n, 0);
    auto& visited = work.visited; visited.assign(n, 0);
    for (std::size_t y = 0; y < roi.height; ++y)
        for (std::size_t x = 0; x < roi.width; ++x) {
            const auto u = roi.x + x, v = roi.y + y;
            const double value = depth.at_m(u, v);
            if ((!mask || mask->contains(u, v)) && std::isfinite(value) &&
                value > config.min_depth_m && value <= config.max_depth_m)
                z[y * roi.width + x] = value;
        }
    std::vector<DepthRegion> regions;
    auto& queue = work.queue;
    constexpr int dx[8]{-1,1,0,0,-1,1,-1,1}, dy[8]{0,0,-1,1,-1,-1,1,1};
    for (std::size_t root = 0; root < n; ++root) {
        if (visited[root] || z[root] == 0) continue;
        queue.clear(); queue.push_back(root); visited[root] = 1;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const auto p = queue[head];
            const auto x = static_cast<int>(p % roi.width), y = static_cast<int>(p / roi.width);
            for (int d = 0; d < config.connectivity; ++d) {
                const int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || ny < 0 || nx >= static_cast<int>(roi.width) || ny >= static_cast<int>(roi.height)) continue;
                const auto q = static_cast<std::size_t>(ny) * roi.width + static_cast<std::size_t>(nx);
                if (visited[q] || z[q] == 0 || std::abs(z[p] - z[q]) >
                    config.absolute_jump_m + config.relative_jump * std::min(z[p], z[q])) continue;
                if (!connected({roi.x + static_cast<std::size_t>(x),roi.y + static_cast<std::size_t>(y)},
                                               {roi.x + static_cast<std::size_t>(nx),roi.y + static_cast<std::size_t>(ny)})) continue;
                visited[q] = 1; queue.push_back(q);
            }
        }
        if (queue.size() < config.min_pixels) continue;
        DepthRegion region;
        region.label = regions.size() + 1;
        auto& values = work.values; values.clear();
        values.reserve(queue.size()); region.pixels.reserve(queue.size());
        std::size_t xmin = depth.width(), xmax = 0, ymin = depth.height(), ymax = 0;
        double mean_u = 0, mean_v = 0;
        for (auto index : queue) {
            const Pixel pixel{roi.x + index % roi.width, roi.y + index / roi.width};
            region.pixels.push_back(pixel); values.push_back(z[index]);
            const double count = static_cast<double>(values.size());
            mean_u += (static_cast<double>(pixel.u) - mean_u) / count;
            mean_v += (static_cast<double>(pixel.v) - mean_v) / count;
            xmin = std::min(xmin, pixel.u); xmax = std::max(xmax, pixel.u);
            ymin = std::min(ymin, pixel.v); ymax = std::max(ymax, pixel.v);
        }
        region.bounds = {xmin, ymin, xmax - xmin + 1, ymax - ymin + 1};
        bool histogram_used = false;
        if constexpr (std::is_same_v<T,std::uint16_t>) {
            if (values.size() >= 4096) {
                // Exact raw-code frequencies; no metric depth quantization. Sorting
                // distinct codes replaces sorting hundreds of thousands of pixels.
                if(work.histogram.empty())work.histogram.assign(65536,0);
                for(auto code:work.active_bins)work.histogram[code]=0;
                work.active_bins.clear();
                for(auto pixel:region.pixels){
                    const auto code=depth.raw_at(pixel.u,pixel.v);
                    if(work.histogram[code]==0)work.active_bins.push_back(code);
                    ++work.histogram[code];
                }
                std::sort(work.active_bins.begin(),work.active_bins.end());
                const auto lower=(values.size()-1)/2,upper=values.size()/2;
                std::size_t cumulative=0;double lo=0,hi=0;
                for(auto code:work.active_bins){
                    const auto next=cumulative+work.histogram[code];
                    if(cumulative<=lower&&lower<next)lo=region_detail::metric_code(code,depth.scale_to_m());
                    if(cumulative<=upper&&upper<next){hi=region_detail::metric_code(code,depth.scale_to_m());break;}
                    cumulative=next;
                }
                region.depth_m=values.size()%2?hi:lo/2+hi/2;
                work.weighted.clear();work.weighted.reserve(work.active_bins.size());
                for(auto code:work.active_bins){
                    // Match at_m()'s rounded metric depth before subtracting: a
                    // fused multiply-subtract would change last-bit MAD values.
                    const double metric=region_detail::metric_code(code,depth.scale_to_m());
                    work.weighted.push_back({std::abs(metric-region.depth_m),work.histogram[code]});
                }
                std::sort(work.weighted.begin(),work.weighted.end(),[](auto a,auto b){return a.value<b.value;});
                cumulative=0;
                for(auto value:work.weighted){
                    const auto next=cumulative+value.count;
                    if(cumulative<=lower&&lower<next)lo=value.value;
                    if(cumulative<=upper&&upper<next){hi=value.value;break;}
                    cumulative=next;
                }
                region.mad_m=values.size()%2?hi:lo/2+hi/2;histogram_used=true;
            }
        }
        if(!histogram_used){
            auto& deviations = work.scratch;
            deviations.assign(values.begin(), values.end());
            region.depth_m = region_detail::median_inplace(deviations);
            // Preserve values' BFS order: it maps the representative pixel to its depth.
            for (std::size_t i = 0; i < values.size(); ++i)
                deviations[i] = std::abs(values[i] - region.depth_m);
            region.mad_m = region_detail::median_inplace(deviations);
        }
        double best_distance = std::numeric_limits<double>::infinity();
        std::size_t best = 0;
        for (std::size_t i = 0; i < region.pixels.size(); ++i) {
            const double du = static_cast<double>(region.pixels[i].u) - mean_u;
            const double dv = static_cast<double>(region.pixels[i].v) - mean_v;
            const double distance = du * du + dv * dv;
            if (distance < best_distance) { best_distance = distance; best = i; }
        }
        region.representative_pixel = region.pixels[best];
        region.point_m = backproject(k, static_cast<double>(region.representative_pixel.u),
                                    static_cast<double>(region.representative_pixel.v), values[best]);
        regions.push_back(std::move(region));
    }
    return regions;
}
} // namespace operator_lab::rgbd
