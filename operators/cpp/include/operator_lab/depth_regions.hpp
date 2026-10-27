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
inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : values[n / 2 - 1] / 2 + values[n / 2] / 2;
}
}

// Local depth-jump connectivity is transitive: a ramp can form one component.
// No person detection, semantic ground removal or automatic RGB registration.
template<class T>
inline std::vector<DepthRegion> extract_depth_regions(const DepthView<T>& depth,
    const Intrinsics& k, Rect roi, const DepthRegionConfig& config = {},
    std::optional<MaskView> mask = std::nullopt,
    const std::function<bool(Pixel,Pixel)>& can_connect = {}) {
    validate(k); validate(config); validate(roi, depth.width(), depth.height());
    if (k.width != depth.width() || k.height != depth.height() ||
        (mask && (mask->width() != depth.width() || mask->height() != depth.height())))
        throw std::invalid_argument("region input dimensions differ");
    const std::size_t n = roi.width * roi.height;
    std::vector<double> z(n, 0);
    std::vector<std::uint8_t> visited(n, 0);
    for (std::size_t y = 0; y < roi.height; ++y)
        for (std::size_t x = 0; x < roi.width; ++x) {
            const auto u = roi.x + x, v = roi.y + y;
            const double value = depth.at_m(u, v);
            if ((!mask || mask->contains(u, v)) && std::isfinite(value) &&
                value > config.min_depth_m && value <= config.max_depth_m)
                z[y * roi.width + x] = value;
        }
    std::vector<DepthRegion> regions;
    std::vector<std::size_t> queue;
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
                if (can_connect && !can_connect({roi.x + p % roi.width,roi.y + p / roi.width},
                                                {roi.x + q % roi.width,roi.y + q / roi.width})) continue;
                visited[q] = 1; queue.push_back(q);
            }
        }
        if (queue.size() < config.min_pixels) continue;
        DepthRegion region;
        region.label = regions.size() + 1;
        std::vector<double> values;
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
        region.depth_m = region_detail::median(values);
        auto deviations = values;
        for (auto& value : deviations) value = std::abs(value - region.depth_m);
        region.mad_m = region_detail::median(deviations);
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
